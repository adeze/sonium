//! Incident field computation
//!
//! Computes incident acoustic fields for BEM excitation sources:
//! - Plane waves
//! - Point sources (monopoles)
//!
//! These are used to form the right-hand side of the BEM system.

use num_complex::Complex64;
use std::f64::consts::PI;
use std::fmt;

use crate::core::types::PhysicsParams;

/// Incident field source type
#[derive(Debug, Clone)]
pub enum IncidentField {
    /// Plane wave: p = A * exp(i k·x)
    PlaneWave {
        /// Direction of propagation (unit vector)
        direction: [f64; 3],
        /// Complex amplitude
        amplitude: Complex64,
    },

    /// Point source (monopole): p = A * exp(ikr) / (4πr)
    PointSource {
        /// Source position
        position: [f64; 3],
        /// Source strength (volume velocity amplitude)
        strength: Complex64,
    },

    /// Multiple plane waves
    MultiplePlaneWaves(Vec<([f64; 3], Complex64)>),

    /// Multiple point sources
    MultiplePointSources(Vec<([f64; 3], Complex64)>),
}

/// Validation failure for a user-provided incident field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncidentFieldError {
    /// A coordinate, direction component, or complex source value is not finite.
    NonFinite {
        /// Field family containing the invalid value.
        kind: &'static str,
        /// Source index for a multi-source field, if applicable.
        index: Option<usize>,
        /// Coordinate/component index for the invalid value, if applicable.
        component: Option<usize>,
    },
}

impl fmt::Display for IncidentFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite {
                kind,
                index,
                component,
            } => write!(
                formatter,
                "{kind} contains a non-finite value{}{}",
                index.map_or_else(String::new, |value| format!(" at index {value}")),
                component.map_or_else(String::new, |value| format!(" component {value}")),
            ),
        }
    }
}

impl std::error::Error for IncidentFieldError {}

impl IncidentField {
    /// Validate all user-provided coordinates, directions, and amplitudes.
    ///
    /// Constructors normalize plane-wave directions, but the enum variants are
    /// public for file/configuration adapters and therefore must be checked at
    /// the high-level solver boundary before entering numerical kernels.
    pub fn validate(&self) -> Result<(), IncidentFieldError> {
        fn finite_complex(value: Complex64) -> bool {
            value.re.is_finite() && value.im.is_finite()
        }

        fn finite_vector(values: &[f64; 3]) -> Option<usize> {
            values.iter().position(|value| !value.is_finite())
        }

        match self {
            Self::PlaneWave {
                direction,
                amplitude,
            } => {
                if let Some(component) = finite_vector(direction) {
                    return Err(IncidentFieldError::NonFinite {
                        kind: "plane-wave direction",
                        index: None,
                        component: Some(component),
                    });
                }
                if !finite_complex(*amplitude) {
                    return Err(IncidentFieldError::NonFinite {
                        kind: "plane-wave amplitude",
                        index: None,
                        component: None,
                    });
                }
            }
            Self::PointSource { position, strength } => {
                if let Some(component) = finite_vector(position) {
                    return Err(IncidentFieldError::NonFinite {
                        kind: "point-source position",
                        index: None,
                        component: Some(component),
                    });
                }
                if !finite_complex(*strength) {
                    return Err(IncidentFieldError::NonFinite {
                        kind: "point-source strength",
                        index: None,
                        component: None,
                    });
                }
            }
            Self::MultiplePlaneWaves(waves) => {
                for (index, (direction, amplitude)) in waves.iter().enumerate() {
                    if let Some(component) = finite_vector(direction) {
                        return Err(IncidentFieldError::NonFinite {
                            kind: "plane-wave direction",
                            index: Some(index),
                            component: Some(component),
                        });
                    }
                    if !finite_complex(*amplitude) {
                        return Err(IncidentFieldError::NonFinite {
                            kind: "plane-wave amplitude",
                            index: Some(index),
                            component: None,
                        });
                    }
                }
            }
            Self::MultiplePointSources(sources) => {
                for (index, (position, strength)) in sources.iter().enumerate() {
                    if let Some(component) = finite_vector(position) {
                        return Err(IncidentFieldError::NonFinite {
                            kind: "point-source position",
                            index: Some(index),
                            component: Some(component),
                        });
                    }
                    if !finite_complex(*strength) {
                        return Err(IncidentFieldError::NonFinite {
                            kind: "point-source strength",
                            index: Some(index),
                            component: None,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Create a plane wave with unit amplitude traveling in +z direction
    ///
    /// This matches the standard convention for Mie scattering theory where
    /// the incident wave travels toward +z (positive z direction).
    pub fn plane_wave_z() -> Self {
        IncidentField::PlaneWave {
            direction: [0.0, 0.0, 1.0],
            amplitude: Complex64::new(1.0, 0.0),
        }
    }

    /// Create a plane wave with unit amplitude traveling in -z direction
    pub fn plane_wave_neg_z() -> Self {
        IncidentField::PlaneWave {
            direction: [0.0, 0.0, -1.0],
            amplitude: Complex64::new(1.0, 0.0),
        }
    }

    /// Create a plane wave with specified direction and amplitude
    pub fn plane_wave(direction: [f64; 3], amplitude: f64) -> Self {
        // Normalize direction
        let len = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        let dir = if len > 1e-10 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, 0.0, -1.0]
        };

        IncidentField::PlaneWave {
            direction: dir,
            amplitude: Complex64::new(amplitude, 0.0),
        }
    }

    /// Create a point source at given position
    pub fn point_source(position: [f64; 3], strength: f64) -> Self {
        IncidentField::PointSource {
            position,
            strength: Complex64::new(strength, 0.0),
        }
    }

    /// Evaluate incident pressure at given points
    ///
    /// # Arguments
    /// * `points` - Evaluation points (N × 3 array)
    /// * `physics` - Physical parameters (contains wave number k)
    ///
    /// # Returns
    /// Complex pressure values at each point
    pub fn evaluate_pressure(
        &self,
        points: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<Complex64> {
        let n = points.len();
        let k = physics.wave_number;
        let phase_sign = physics.harmonic_factor;
        let mut pressure = vec![Complex64::new(0.0, 0.0); n];

        match self {
            IncidentField::PlaneWave {
                direction,
                amplitude,
            } => {
                for i in 0..n {
                    // k·x = k * (d·x)
                    let kdotx = phase_sign
                        * k
                        * (direction[0] * points[i][0]
                            + direction[1] * points[i][1]
                            + direction[2] * points[i][2]);

                    // p = A * exp(i k·x)
                    pressure[i] = *amplitude * sonium_fastmath::exp_i(kdotx);
                }
            }

            IncidentField::PointSource { position, strength } => {
                for i in 0..n {
                    let dx = points[i][0] - position[0];
                    let dy = points[i][1] - position[1];
                    let dz = points[i][2] - position[2];
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();

                    if r > 1e-10 {
                        let kr = phase_sign * k * r;
                        // p = S * exp(ikr) / (4πr)
                        let green = sonium_fastmath::exp_i(kr) / (4.0 * PI * r);
                        pressure[i] = *strength * green;
                    }
                }
            }

            IncidentField::MultiplePlaneWaves(waves) => {
                for (direction, amplitude) in waves {
                    for i in 0..n {
                        let kdotx = phase_sign
                            * k
                            * (direction[0] * points[i][0]
                                + direction[1] * points[i][1]
                                + direction[2] * points[i][2]);
                        pressure[i] += *amplitude * sonium_fastmath::exp_i(kdotx);
                    }
                }
            }

            IncidentField::MultiplePointSources(sources) => {
                for (position, strength) in sources {
                    for i in 0..n {
                        let dx = points[i][0] - position[0];
                        let dy = points[i][1] - position[1];
                        let dz = points[i][2] - position[2];
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();

                        if r > 1e-10 {
                            let kr = phase_sign * k * r;
                            let green = sonium_fastmath::exp_i(kr) / (4.0 * PI * r);
                            pressure[i] += *strength * green;
                        }
                    }
                }
            }
        }

        pressure
    }

    /// Evaluate incident velocity (∂p/∂n) at given points with normals
    ///
    /// # Arguments
    /// * `points` - Evaluation points (N × 3 array)
    /// * `normals` - Unit normal vectors at each point (N × 3 array)
    /// * `physics` - Physical parameters
    ///
    /// # Returns
    /// Complex normal velocity values (actually ∂p/∂n, not v_n)
    pub fn evaluate_normal_derivative(
        &self,
        points: &[[f64; 3]],
        normals: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<Complex64> {
        let n = points.len();
        let k = physics.wave_number;
        let phase_sign = physics.harmonic_factor;
        let mut dpdn = vec![Complex64::new(0.0, 0.0); n];

        match self {
            IncidentField::PlaneWave {
                direction,
                amplitude,
            } => {
                for i in 0..n {
                    let kdotx = phase_sign
                        * k
                        * (direction[0] * points[i][0]
                            + direction[1] * points[i][1]
                            + direction[2] * points[i][2]);

                    // ∂p/∂n = ik (d·n) p
                    let kdotn = phase_sign
                        * k
                        * (direction[0] * normals[i][0]
                            + direction[1] * normals[i][1]
                            + direction[2] * normals[i][2]);

                    let p = *amplitude * sonium_fastmath::exp_i(kdotx);
                    dpdn[i] = Complex64::new(0.0, kdotn) * p;
                }
            }

            IncidentField::PointSource { position, strength } => {
                for i in 0..n {
                    let dx = points[i][0] - position[0];
                    let dy = points[i][1] - position[1];
                    let dz = points[i][2] - position[2];
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();

                    if r > 1e-10 {
                        let kr = phase_sign * k * r;
                        let _r2 = r * r;

                        // ∂G/∂r = (ik - 1/r) * G
                        let exp_ikr = sonium_fastmath::exp_i(kr);
                        let g = exp_ikr / (4.0 * PI * r);
                        let dgdr = (Complex64::new(0.0, phase_sign * k)
                            - Complex64::new(1.0 / r, 0.0))
                            * g;

                        // ∂r/∂n = (x-x0)·n / r
                        let drdn =
                            (dx * normals[i][0] + dy * normals[i][1] + dz * normals[i][2]) / r;

                        dpdn[i] = *strength * dgdr * drdn;
                    }
                }
            }

            IncidentField::MultiplePlaneWaves(waves) => {
                for (direction, amplitude) in waves {
                    for i in 0..n {
                        let kdotx = phase_sign
                            * k
                            * (direction[0] * points[i][0]
                                + direction[1] * points[i][1]
                                + direction[2] * points[i][2]);

                        let kdotn = phase_sign
                            * k
                            * (direction[0] * normals[i][0]
                                + direction[1] * normals[i][1]
                                + direction[2] * normals[i][2]);

                        let p = *amplitude * sonium_fastmath::exp_i(kdotx);
                        dpdn[i] += Complex64::new(0.0, kdotn) * p;
                    }
                }
            }

            IncidentField::MultiplePointSources(sources) => {
                for (position, strength) in sources {
                    for i in 0..n {
                        let dx = points[i][0] - position[0];
                        let dy = points[i][1] - position[1];
                        let dz = points[i][2] - position[2];
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();

                        if r > 1e-10 {
                            let kr = phase_sign * k * r;
                            let exp_ikr = sonium_fastmath::exp_i(kr);
                            let g = exp_ikr / (4.0 * PI * r);
                            let dgdr = (Complex64::new(0.0, phase_sign * k)
                                - Complex64::new(1.0 / r, 0.0))
                                * g;
                            let drdn =
                                (dx * normals[i][0] + dy * normals[i][1] + dz * normals[i][2]) / r;

                            dpdn[i] += *strength * dgdr * drdn;
                        }
                    }
                }
            }
        }

        dpdn
    }

    /// Analytic derivative of incident pressure with respect to wavenumber.
    ///
    /// The source geometry and complex amplitudes remain fixed.  This is the
    /// material/frequency derivative needed by frequency-sweep sensitivities;
    /// callers convert it to a frequency or sound-speed derivative using the
    /// corresponding derivative of [`PhysicsParams::wave_number`].
    pub fn evaluate_pressure_wavenumber_derivative(
        &self,
        points: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<Complex64> {
        let k = physics.wave_number;
        let phase_sign = physics.harmonic_factor;
        let mut derivative = vec![Complex64::new(0.0, 0.0); points.len()];

        let accumulate = |point: &[f64; 3],
                          source_position: Option<&[f64; 3]>,
                          amplitude: Complex64,
                          output: &mut Complex64| {
            let distance = if let Some(position) = source_position {
                let dx = point[0] - position[0];
                let dy = point[1] - position[1];
                let dz = point[2] - position[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            } else {
                point[0].abs() + point[1].abs() + point[2].abs()
            };
            if source_position.is_some() && distance <= 1e-10 {
                return;
            }
            let phase_distance = if let Some(position) = source_position {
                let dx = point[0] - position[0];
                let dy = point[1] - position[1];
                let dz = point[2] - position[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            } else {
                distance
            };
            let scale = if source_position.is_some() {
                1.0 / (4.0 * PI * phase_distance)
            } else {
                1.0
            };
            *output += amplitude
                * sonium_fastmath::exp_i(phase_sign * k * phase_distance)
                * Complex64::new(0.0, phase_sign * phase_distance * scale);
        };

        match self {
            Self::PlaneWave {
                direction,
                amplitude,
            } => {
                for (point, value) in points.iter().zip(&mut derivative) {
                    let projection =
                        direction[0] * point[0] + direction[1] * point[1] + direction[2] * point[2];
                    *value = *amplitude
                        * sonium_fastmath::exp_i(phase_sign * k * projection)
                        * Complex64::new(0.0, phase_sign * projection);
                }
            }
            Self::PointSource { position, strength } => {
                for (point, value) in points.iter().zip(&mut derivative) {
                    accumulate(point, Some(position), *strength, value);
                }
            }
            Self::MultiplePlaneWaves(waves) => {
                for (direction, amplitude) in waves {
                    for (point, value) in points.iter().zip(&mut derivative) {
                        let projection = direction[0] * point[0]
                            + direction[1] * point[1]
                            + direction[2] * point[2];
                        *value += *amplitude
                            * sonium_fastmath::exp_i(phase_sign * k * projection)
                            * Complex64::new(0.0, phase_sign * projection);
                    }
                }
            }
            Self::MultiplePointSources(sources) => {
                for (position, strength) in sources {
                    for (point, value) in points.iter().zip(&mut derivative) {
                        accumulate(point, Some(position), *strength, value);
                    }
                }
            }
        }

        derivative
    }

    /// Analytic derivative of incident pressure with respect to point-source
    /// position.  Plane-wave contributions are zero; multiple point sources
    /// are accumulated independently.  The returned array is ordered as
    /// `[d p / d x_source, d p / d y_source, d p / d z_source]`.
    pub fn evaluate_pressure_source_position_derivative(
        &self,
        points: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<[Complex64; 3]> {
        let mut derivative = vec![[Complex64::new(0.0, 0.0); 3]; points.len()];
        match self {
            Self::PointSource { position, strength } => {
                for (point, value) in points.iter().zip(&mut derivative) {
                    add_point_source_position_derivative(
                        value, point, position, *strength, physics,
                    );
                }
            }
            Self::MultiplePointSources(sources) => {
                for (position, strength) in sources {
                    for (point, value) in points.iter().zip(&mut derivative) {
                        add_point_source_position_derivative(
                            value, point, position, *strength, physics,
                        );
                    }
                }
            }
            Self::PlaneWave { .. } | Self::MultiplePlaneWaves(_) => {}
        }
        derivative
    }

    /// Evaluate incident pressure derivative with respect to an evaluation
    /// point position. This is the geometry-translation derivative needed
    /// when a rigid surface is translated while the incident source remains
    /// fixed in world coordinates.
    ///
    /// For point sources this is the negative of the source-position
    /// derivative; plane-wave and multiple-plane-wave contributions are
    /// evaluated directly.
    pub fn evaluate_pressure_point_position_derivative(
        &self,
        points: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<[Complex64; 3]> {
        let mut derivative = vec![[Complex64::new(0.0, 0.0); 3]; points.len()];
        let k = physics.wave_number;
        let phase_sign = physics.harmonic_factor;
        let plane_wave = |point: &[f64; 3], direction: &[f64; 3], amplitude: Complex64| {
            let projection =
                direction[0] * point[0] + direction[1] * point[1] + direction[2] * point[2];
            let pressure = amplitude * sonium_fastmath::exp_i(phase_sign * k * projection);
            [
                pressure * Complex64::new(0.0, phase_sign * k * direction[0]),
                pressure * Complex64::new(0.0, phase_sign * k * direction[1]),
                pressure * Complex64::new(0.0, phase_sign * k * direction[2]),
            ]
        };
        match self {
            Self::PlaneWave {
                direction,
                amplitude,
            } => {
                for (point, value) in points.iter().zip(&mut derivative) {
                    *value = plane_wave(point, direction, *amplitude);
                }
            }
            Self::MultiplePlaneWaves(waves) => {
                for (direction, amplitude) in waves {
                    for (point, value) in points.iter().zip(&mut derivative) {
                        let contribution = plane_wave(point, direction, *amplitude);
                        for axis in 0..3 {
                            value[axis] += contribution[axis];
                        }
                    }
                }
            }
            Self::PointSource { .. } | Self::MultiplePointSources(_) => {
                let source_derivative =
                    self.evaluate_pressure_source_position_derivative(points, physics);
                for (value, source_value) in derivative.iter_mut().zip(source_derivative) {
                    for axis in 0..3 {
                        value[axis] = -source_value[axis];
                    }
                }
            }
        }
        derivative
    }

    /// Evaluate the incident normal-velocity derivative with respect to a
    /// point-source position.  The returned entries are ordered as
    /// `[d(∂p/∂n)/dx_source, d(∂p/∂n)/dy_source,
    /// d(∂p/∂n)/dz_source]` for each point/normal pair.
    ///
    /// Plane-wave contributions are zero; multiple point sources are
    /// accumulated independently.  This is the analytic derivative of the
    /// same Green-function expression used by [`Self::evaluate_normal_derivative`].
    pub fn evaluate_normal_source_position_derivative(
        &self,
        points: &[[f64; 3]],
        normals: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<[Complex64; 3]> {
        assert_eq!(
            points.len(),
            normals.len(),
            "points/normals length mismatch"
        );
        let mut derivative = vec![[Complex64::new(0.0, 0.0); 3]; points.len()];
        match self {
            Self::PointSource { position, strength } => {
                for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative) {
                    add_point_source_normal_position_derivative(
                        value, point, normal, position, *strength, physics,
                    );
                }
            }
            Self::MultiplePointSources(sources) => {
                for (position, strength) in sources {
                    for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative)
                    {
                        add_point_source_normal_position_derivative(
                            value, point, normal, position, *strength, physics,
                        );
                    }
                }
            }
            Self::PlaneWave { .. } | Self::MultiplePlaneWaves(_) => {}
        }
        derivative
    }

    /// Evaluate the incident normal derivative with respect to an evaluation
    /// point position while holding the supplied normal fixed. Point-source
    /// terms are obtained by the exact sign reversal of the source-position
    /// derivative; plane-wave terms are differentiated analytically.
    pub fn evaluate_normal_point_position_derivative(
        &self,
        points: &[[f64; 3]],
        normals: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<[Complex64; 3]> {
        assert_eq!(
            points.len(),
            normals.len(),
            "points/normals length mismatch"
        );
        let mut derivative = vec![[Complex64::new(0.0, 0.0); 3]; points.len()];
        let k = physics.wave_number;
        let phase_sign = physics.harmonic_factor;
        let plane_wave =
            |point: &[f64; 3], normal: &[f64; 3], direction: &[f64; 3], amplitude: Complex64| {
                let projection =
                    direction[0] * point[0] + direction[1] * point[1] + direction[2] * point[2];
                let normal_projection =
                    direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2];
                let normal_pressure = amplitude
                    * sonium_fastmath::exp_i(phase_sign * k * projection)
                    * Complex64::new(0.0, phase_sign * k * normal_projection);
                [
                    normal_pressure * Complex64::new(0.0, phase_sign * k * direction[0]),
                    normal_pressure * Complex64::new(0.0, phase_sign * k * direction[1]),
                    normal_pressure * Complex64::new(0.0, phase_sign * k * direction[2]),
                ]
            };
        match self {
            Self::PlaneWave {
                direction,
                amplitude,
            } => {
                for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative) {
                    *value = plane_wave(point, normal, direction, *amplitude);
                }
            }
            Self::MultiplePlaneWaves(waves) => {
                for (direction, amplitude) in waves {
                    for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative)
                    {
                        let contribution = plane_wave(point, normal, direction, *amplitude);
                        for axis in 0..3 {
                            value[axis] += contribution[axis];
                        }
                    }
                }
            }
            Self::PointSource { .. } | Self::MultiplePointSources(_) => {
                let source_derivative =
                    self.evaluate_normal_source_position_derivative(points, normals, physics);
                for (value, source_value) in derivative.iter_mut().zip(source_derivative) {
                    for axis in 0..3 {
                        value[axis] = -source_value[axis];
                    }
                }
            }
        }
        derivative
    }

    /// Analytic derivative of the incident normal derivative with respect to
    /// wavenumber.  The returned quantity is `d/dk (∂p/∂n)`, not particle
    /// velocity; this matches [`Self::evaluate_normal_derivative`].
    pub fn evaluate_normal_wavenumber_derivative(
        &self,
        points: &[[f64; 3]],
        normals: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Vec<Complex64> {
        assert_eq!(
            points.len(),
            normals.len(),
            "points/normals length mismatch"
        );
        let k = physics.wave_number;
        let phase_sign = physics.harmonic_factor;
        let mut derivative = vec![Complex64::new(0.0, 0.0); points.len()];

        let accumulate_plane =
            |point: &[f64; 3], normal: &[f64; 3], direction: &[f64; 3], amplitude: Complex64| {
                let projection =
                    direction[0] * point[0] + direction[1] * point[1] + direction[2] * point[2];
                let normal_projection =
                    direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2];
                let p = amplitude * sonium_fastmath::exp_i(phase_sign * k * projection);
                p * (Complex64::new(0.0, phase_sign * normal_projection)
                    + Complex64::new(0.0, phase_sign * k * normal_projection)
                        * Complex64::new(0.0, phase_sign * projection))
            };

        match self {
            Self::PlaneWave {
                direction,
                amplitude,
            } => {
                for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative) {
                    *value = accumulate_plane(point, normal, direction, *amplitude);
                }
            }
            Self::MultiplePlaneWaves(waves) => {
                for (direction, amplitude) in waves {
                    for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative)
                    {
                        *value += accumulate_plane(point, normal, direction, *amplitude);
                    }
                }
            }
            Self::PointSource { position, strength } => {
                for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative) {
                    *value += point_source_normal_wavenumber_derivative(
                        point, normal, position, *strength, k, phase_sign,
                    );
                }
            }
            Self::MultiplePointSources(sources) => {
                for (position, strength) in sources {
                    for ((point, normal), value) in points.iter().zip(normals).zip(&mut derivative)
                    {
                        *value += point_source_normal_wavenumber_derivative(
                            point, normal, position, *strength, k, phase_sign,
                        );
                    }
                }
            }
        }

        derivative
    }

    /// Compute the right-hand side vector for BEM
    ///
    /// For exterior Neumann problem (rigid scatterer) using DIRECT formulation
    /// where the unknown is the total surface pressure p:
    ///
    /// The RHS comes from the incident field contribution to the integral equation.
    /// From NumCalc NC_IncidentWaveRHS:
    ///   RHS = -(γ*p_inc + β*τ*∂p_inc/∂n)
    ///
    /// For exterior problems: τ = +1, γ = 1
    /// For interior problems: τ = -1
    pub fn compute_rhs(
        &self,
        element_centers: &[[f64; 3]],
        element_normals: &[[f64; 3]],
        physics: &PhysicsParams,
        use_burton_miller: bool,
    ) -> Vec<Complex64> {
        if use_burton_miller {
            let beta = physics.burton_miller_beta();
            self.compute_rhs_with_beta(element_centers, element_normals, physics, beta)
        } else {
            // CBIE only: RHS = -γ*p_inc
            let gamma = Complex64::new(physics.gamma(), 0.0);
            let p_inc = self.evaluate_pressure(element_centers, physics);
            p_inc.iter().map(|p| -gamma * p).collect()
        }
    }

    /// Compute RHS with custom Burton-Miller coupling parameter
    ///
    /// From NumCalc NC_IncidentWaveRHS:
    ///   RHS = -(γ*p_inc + β*τ*∂p_inc/∂n)
    ///
    /// This formula matches the C++ implementation for proper BEM assembly.
    pub fn compute_rhs_with_beta(
        &self,
        element_centers: &[[f64; 3]],
        element_normals: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
    ) -> Vec<Complex64> {
        // Get incident pressure at collocation points
        let p_inc = self.evaluate_pressure(element_centers, physics);

        // Get incident velocity (normal derivative) at collocation points
        let dpdn = self.evaluate_normal_derivative(element_centers, element_normals, physics);

        // Parameters from physics
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);

        // Burton-Miller RHS: -(γ*p_inc + β*τ*∂p_inc/∂n)
        // This matches NC_IncidentWaveRHS: zrc -= zg*Gama3 + zv*(zBta3*Tao_)
        let n = element_centers.len();
        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            rhs[i] = -(gamma * p_inc[i] + beta * tau * dpdn[i]);
        }

        rhs
    }

    /// Compute the analytic Burton--Miller RHS derivative with respect to a
    /// point-source position.  The output is one three-component derivative
    /// per panel and follows the production RHS convention
    /// `-(gamma*p_inc + beta*tau*∂p_inc/∂n)`.
    pub fn compute_rhs_source_position_derivative(
        &self,
        element_centers: &[[f64; 3]],
        element_normals: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
    ) -> Vec<[Complex64; 3]> {
        assert_eq!(
            element_centers.len(),
            element_normals.len(),
            "element centers/normals length mismatch"
        );
        let pressure = self.evaluate_pressure_source_position_derivative(element_centers, physics);
        let normal = self.evaluate_normal_source_position_derivative(
            element_centers,
            element_normals,
            physics,
        );
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        pressure
            .into_iter()
            .zip(normal)
            .map(|(pressure, normal)| {
                std::array::from_fn(|axis| -(gamma * pressure[axis] + beta * tau * normal[axis]))
            })
            .collect()
    }

    /// Compute the Burton--Miller RHS derivative with respect to a rigid
    /// translation of every collocation point. The operator itself is
    /// translation invariant, so this is the complete fixed-topology
    /// geometry derivative for a global surface translation.
    pub fn compute_rhs_geometry_translation_derivative(
        &self,
        element_centers: &[[f64; 3]],
        element_normals: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
    ) -> Vec<[Complex64; 3]> {
        assert_eq!(
            element_centers.len(),
            element_normals.len(),
            "element centers/normals length mismatch"
        );
        let pressure = self.evaluate_pressure_point_position_derivative(element_centers, physics);
        let normal = self.evaluate_normal_point_position_derivative(
            element_centers,
            element_normals,
            physics,
        );
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        pressure
            .into_iter()
            .zip(normal)
            .map(|(pressure, normal)| {
                [
                    -(gamma * pressure[0] + beta * tau * normal[0]),
                    -(gamma * pressure[1] + beta * tau * normal[1]),
                    -(gamma * pressure[2] + beta * tau * normal[2]),
                ]
            })
            .collect()
    }

    /// Analytic derivative of the Burton--Miller incident RHS with respect to
    /// wavenumber.
    ///
    /// The derivative includes both the incident-field terms and the explicit
    /// derivative of the traditional `beta(k)` coupling.  Use
    /// [`Self::compute_rhs_wavenumber_derivative_with_beta`] when a bounded or
    /// otherwise custom beta schedule is used.
    pub fn compute_rhs_wavenumber_derivative(
        &self,
        element_centers: &[[f64; 3]],
        element_normals: &[[f64; 3]],
        physics: &PhysicsParams,
        use_burton_miller: bool,
    ) -> Vec<Complex64> {
        let (beta, beta_derivative) = if use_burton_miller {
            (
                physics.burton_miller_beta(),
                physics.burton_miller_beta_wavenumber_derivative(),
            )
        } else {
            (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0))
        };
        self.compute_rhs_wavenumber_derivative_with_beta(
            element_centers,
            element_normals,
            physics,
            beta,
            beta_derivative,
        )
    }

    /// Analytic derivative of the incident RHS for an explicitly supplied
    /// `beta(k)` and `d beta/dk` pair.
    pub fn compute_rhs_wavenumber_derivative_with_beta(
        &self,
        element_centers: &[[f64; 3]],
        element_normals: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        beta_derivative: Complex64,
    ) -> Vec<Complex64> {
        assert_eq!(
            element_centers.len(),
            element_normals.len(),
            "element centers/normals length mismatch"
        );
        let pressure_derivative =
            self.evaluate_pressure_wavenumber_derivative(element_centers, physics);
        let normal_derivative =
            self.evaluate_normal_wavenumber_derivative(element_centers, element_normals, physics);
        let dpdn = self.evaluate_normal_derivative(element_centers, element_normals, physics);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);

        pressure_derivative
            .into_iter()
            .zip(normal_derivative)
            .zip(dpdn)
            .map(|((pressure, normal), dpdn)| {
                -(gamma * pressure + tau * (beta_derivative * dpdn + beta * normal))
            })
            .collect()
    }
}

fn point_source_normal_wavenumber_derivative(
    point: &[f64; 3],
    normal: &[f64; 3],
    position: &[f64; 3],
    strength: Complex64,
    k: f64,
    phase_sign: f64,
) -> Complex64 {
    let diff = [
        point[0] - position[0],
        point[1] - position[1],
        point[2] - position[2],
    ];
    let r = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    if r <= 1e-10 {
        return Complex64::new(0.0, 0.0);
    }
    let radial_normal = (diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2]) / r;
    let g = sonium_fastmath::exp_i(phase_sign * k * r) / (4.0 * PI * r);
    let dgdr = (Complex64::new(0.0, phase_sign * k) - Complex64::new(1.0 / r, 0.0)) * g;
    let d_dk = Complex64::new(0.0, phase_sign) * g + dgdr * Complex64::new(0.0, phase_sign * r);
    strength * radial_normal * d_dk
}

fn add_point_source_position_derivative(
    output: &mut [Complex64; 3],
    point: &[f64; 3],
    position: &[f64; 3],
    strength: Complex64,
    physics: &PhysicsParams,
) {
    let diff = [
        point[0] - position[0],
        point[1] - position[1],
        point[2] - position[2],
    ];
    let r = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    if r <= 1e-10 {
        return;
    }
    let g =
        sonium_fastmath::exp_i(physics.harmonic_factor * physics.wave_number * r) / (4.0 * PI * r);
    let dgdr = (Complex64::new(0.0, physics.harmonic_factor * physics.wave_number)
        - Complex64::new(1.0 / r, 0.0))
        * g;
    for axis in 0..3 {
        // d r / d source_axis = -(point-source)_axis / r.
        output[axis] -= strength * dgdr * (diff[axis] / r);
    }
}

fn add_point_source_normal_position_derivative(
    output: &mut [Complex64; 3],
    point: &[f64; 3],
    normal: &[f64; 3],
    position: &[f64; 3],
    strength: Complex64,
    physics: &PhysicsParams,
) {
    let diff = [
        point[0] - position[0],
        point[1] - position[1],
        point[2] - position[2],
    ];
    let r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    let r = r2.sqrt();
    if r <= 1e-10 {
        return;
    }

    let unit = [diff[0] / r, diff[1] / r, diff[2] / r];
    let radial_normal = unit[0] * normal[0] + unit[1] * normal[1] + unit[2] * normal[2];
    let g =
        sonium_fastmath::exp_i(physics.harmonic_factor * physics.wave_number * r) / (4.0 * PI * r);
    let ik = Complex64::new(0.0, physics.harmonic_factor * physics.wave_number);
    let inv_r = Complex64::new(1.0 / r, 0.0);
    let dgdr = (ik - inv_r) * g;
    let d2gdr2 =
        (ik * ik - Complex64::new(2.0 / r, 0.0) * ik + Complex64::new(2.0 / (r * r), 0.0)) * g;

    for axis in 0..3 {
        // d/d source_axis [G'(r) (d·n)/r], with d = x - source.
        let geometric = normal[axis] / r - radial_normal * unit[axis] / r;
        output[axis] -= strength * (d2gdr2 * unit[axis] * radial_normal + dgdr * geometric);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_physics(k: f64) -> PhysicsParams {
        let c = 343.0;
        let freq = k * c / (2.0 * PI);
        PhysicsParams::new(freq, c, 1.21, false)
    }

    #[test]
    fn validate_accepts_finite_incident_families() {
        let fields = [
            IncidentField::PlaneWave {
                direction: [0.0, 0.0, 1.0],
                amplitude: Complex64::new(1.0, -0.25),
            },
            IncidentField::PointSource {
                position: [0.1, -0.2, 0.3],
                strength: Complex64::new(-0.4, 0.7),
            },
            IncidentField::MultiplePlaneWaves(vec![([1.0, 0.0, 0.0], Complex64::new(0.2, 0.1))]),
            IncidentField::MultiplePointSources(vec![([0.1, 0.2, 0.3], Complex64::new(0.4, -0.3))]),
        ];
        for field in fields {
            assert_eq!(field.validate(), Ok(()));
        }
    }

    #[test]
    fn validate_rejects_non_finite_coordinates_and_amplitudes() {
        let cases = [
            IncidentField::PlaneWave {
                direction: [f64::NAN, 0.0, 1.0],
                amplitude: Complex64::new(1.0, 0.0),
            },
            IncidentField::PointSource {
                position: [0.0, f64::INFINITY, 0.0],
                strength: Complex64::new(1.0, 0.0),
            },
            IncidentField::MultiplePlaneWaves(vec![(
                [0.0, 0.0, 1.0],
                Complex64::new(f64::NAN, 0.0),
            )]),
            IncidentField::MultiplePointSources(vec![(
                [0.0, 0.0, 1.0],
                Complex64::new(0.0, f64::NEG_INFINITY),
            )]),
        ];
        for field in cases {
            let error = field
                .validate()
                .expect_err("non-finite incident field accepted");
            assert!(error.to_string().contains("non-finite"));
        }
    }

    #[test]
    fn test_plane_wave_on_axis() {
        let incident = IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0);
        let physics = make_physics(1.0);

        // Points on z-axis
        let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];

        let p = incident.evaluate_pressure(&points, &physics);

        // At z=0: exp(0) = 1
        assert!((p[0].re - 1.0).abs() < 1e-10);
        assert!(p[0].im.abs() < 1e-10);

        // At z=1: exp(ik) = exp(i) = cos(1) + i*sin(1)
        // Since direction is +z, kdotx = k*z = 1 at z=1
        assert!((p[1].re - (1.0_f64).cos()).abs() < 1e-10);
        assert!((p[1].im - (1.0_f64).sin()).abs() < 1e-10);
    }

    #[test]
    fn test_point_source_decay() {
        let incident = IncidentField::point_source([0.0, 0.0, 0.0], 1.0);
        let physics = make_physics(1.0);

        // Points at different distances
        let points: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]];

        let p = incident.evaluate_pressure(&points, &physics);

        // Pressure should decay as 1/r
        let ratio_1_2 = p[0].norm() / p[1].norm();
        let ratio_2_4 = p[1].norm() / p[2].norm();

        assert!((ratio_1_2 - 2.0).abs() < 0.1);
        assert!((ratio_2_4 - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_plane_wave_normal_derivative() {
        let incident = IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0);
        let physics = make_physics(1.0);

        // Point with normal pointing in +z direction (along wave direction)
        let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
        let normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]];

        let dpdn = incident.evaluate_normal_derivative(&points, &normals, &physics);

        // ∂p/∂n = ik (d·n) p = ik * (+1) * 1 = +ik
        assert!(dpdn[0].re.abs() < 1e-10);
        assert!((dpdn[0].im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rhs_computation() {
        let incident = IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0);
        let physics = make_physics(1.0);

        let centers: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]];
        let normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]];

        let rhs = incident.compute_rhs(&centers, &normals, &physics, false);

        // RHS = -∂p_inc/∂n, should be non-zero
        assert!(rhs[0].norm() > 0.0);
    }

    #[test]
    fn incident_wavenumber_derivatives_match_central_difference() {
        let points = vec![[0.2, -0.1, 0.7], [1.1, 0.4, -0.3]];
        let normals = vec![[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]];
        let incident = IncidentField::MultiplePointSources(vec![
            ([0.0, 0.0, 0.0], Complex64::new(1.0, 0.2)),
            ([1.0, -0.5, 0.25], Complex64::new(-0.4, 0.7)),
        ]);
        let k = 1.7;
        let step = 1.0e-6;
        let physics = make_physics(k);
        let plus = make_physics(k + step);
        let minus = make_physics(k - step);
        let pressure_derivative =
            incident.evaluate_pressure_wavenumber_derivative(&points, &physics);
        let normal_derivative =
            incident.evaluate_normal_wavenumber_derivative(&points, &normals, &physics);
        let pressure_plus = incident.evaluate_pressure(&points, &plus);
        let pressure_minus = incident.evaluate_pressure(&points, &minus);
        let normal_plus = incident.evaluate_normal_derivative(&points, &normals, &plus);
        let normal_minus = incident.evaluate_normal_derivative(&points, &normals, &minus);

        for (
            ((actual_pressure, actual_normal), (plus_pressure, minus_pressure)),
            (plus_normal, minus_normal),
        ) in pressure_derivative
            .iter()
            .zip(&normal_derivative)
            .zip(pressure_plus.iter().zip(&pressure_minus))
            .zip(normal_plus.iter().zip(&normal_minus))
        {
            let expected_pressure = (*plus_pressure - *minus_pressure) / (2.0 * step);
            let expected_normal = (*plus_normal - *minus_normal) / (2.0 * step);
            assert!((*actual_pressure - expected_pressure).norm() < 1.0e-8);
            assert!((*actual_normal - expected_normal).norm() < 1.0e-8);
        }
    }

    #[test]
    fn burton_miller_rhs_wavenumber_derivative_matches_central_difference() {
        let incident = IncidentField::plane_wave([0.2, -0.3, 0.9], 1.0);
        let centers = vec![[0.2, 0.1, 0.4], [-0.3, 0.7, 0.2]];
        let normals = vec![[0.0, 0.0, 1.0], [0.6, 0.2, 0.77]];
        let k = 1.4;
        let step = 1.0e-6;
        let physics = make_physics(k);
        let plus = make_physics(k + step);
        let minus = make_physics(k - step);
        let analytic =
            incident.compute_rhs_wavenumber_derivative(&centers, &normals, &physics, true);
        let rhs_plus = incident.compute_rhs(&centers, &normals, &plus, true);
        let rhs_minus = incident.compute_rhs(&centers, &normals, &minus, true);
        for ((actual, plus), minus) in analytic.iter().zip(rhs_plus).zip(rhs_minus) {
            let expected = (plus - minus) / (2.0 * step);
            assert!((*actual - expected).norm() < 1.0e-8);
        }
    }

    #[test]
    fn point_source_position_derivative_matches_central_difference() {
        let point = [0.7, -0.2, 1.1];
        let position = [0.1, 0.3, -0.4];
        let strength = Complex64::new(0.8, -0.25);
        let physics = make_physics(1.9);
        let incident = IncidentField::PointSource { position, strength };
        let analytic = incident.evaluate_pressure_source_position_derivative(&[point], &physics);
        let step = 1.0e-6;
        for axis in 0..3 {
            let mut plus_position = position;
            plus_position[axis] += step;
            let mut minus_position = position;
            minus_position[axis] -= step;
            let plus = IncidentField::PointSource {
                position: plus_position,
                strength,
            }
            .evaluate_pressure(&[point], &physics)[0];
            let minus = IncidentField::PointSource {
                position: minus_position,
                strength,
            }
            .evaluate_pressure(&[point], &physics)[0];
            let expected = (plus - minus) / (2.0 * step);
            assert!((analytic[0][axis] - expected).norm() < 1.0e-8);
        }
    }

    #[test]
    fn point_source_normal_position_derivative_matches_central_difference() {
        let point = [0.7, -0.2, 1.1];
        let normal = [0.3, -0.4, 0.8660254037844386];
        let position = [0.1, 0.3, -0.4];
        let strength = Complex64::new(0.8, -0.25);
        let physics = make_physics(1.9);
        let incident = IncidentField::PointSource { position, strength };
        let analytic =
            incident.evaluate_normal_source_position_derivative(&[point], &[normal], &physics);
        let step = 1.0e-6;
        for axis in 0..3 {
            let mut plus_position = position;
            plus_position[axis] += step;
            let mut minus_position = position;
            minus_position[axis] -= step;
            let plus = IncidentField::PointSource {
                position: plus_position,
                strength,
            }
            .evaluate_normal_derivative(&[point], &[normal], &physics)[0];
            let minus = IncidentField::PointSource {
                position: minus_position,
                strength,
            }
            .evaluate_normal_derivative(&[point], &[normal], &physics)[0];
            let expected = (plus - minus) / (2.0 * step);
            assert!((analytic[0][axis] - expected).norm() < 1.0e-7);
        }
    }

    #[test]
    fn plane_wave_source_position_derivatives_are_zero() {
        let incident = IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0);
        let physics = make_physics(1.2);
        let points = [[0.2, -0.1, 0.7]];
        let normals = [[0.0, 0.0, 1.0]];
        assert_eq!(
            incident.evaluate_pressure_source_position_derivative(&points, &physics),
            vec![[Complex64::new(0.0, 0.0); 3]]
        );
        assert_eq!(
            incident.evaluate_normal_source_position_derivative(&points, &normals, &physics),
            vec![[Complex64::new(0.0, 0.0); 3]]
        );
    }

    #[test]
    fn point_position_derivatives_match_evaluation_point_difference() {
        let point = [0.7, -0.2, 1.1];
        let normal = [0.3, -0.4, 0.8660254037844386];
        let position = [0.1, 0.3, -0.4];
        let strength = Complex64::new(0.8, -0.25);
        let physics = make_physics(1.9);
        let incident = IncidentField::PointSource { position, strength };
        let pressure = incident.evaluate_pressure_point_position_derivative(&[point], &physics);
        let normal_derivative =
            incident.evaluate_normal_point_position_derivative(&[point], &[normal], &physics);
        let step = 1.0e-6;
        for axis in 0..3 {
            let mut plus_point = point;
            plus_point[axis] += step;
            let mut minus_point = point;
            minus_point[axis] -= step;
            let plus_pressure = incident.evaluate_pressure(&[plus_point], &physics)[0];
            let minus_pressure = incident.evaluate_pressure(&[minus_point], &physics)[0];
            let expected_pressure = (plus_pressure - minus_pressure) / (2.0 * step);
            assert!((pressure[0][axis] - expected_pressure).norm() < 1.0e-8);
            let plus_normal =
                incident.evaluate_normal_derivative(&[plus_point], &[normal], &physics)[0];
            let minus_normal =
                incident.evaluate_normal_derivative(&[minus_point], &[normal], &physics)[0];
            let expected_normal = (plus_normal - minus_normal) / (2.0 * step);
            assert!((normal_derivative[0][axis] - expected_normal).norm() < 1.0e-7);
        }
    }
}
