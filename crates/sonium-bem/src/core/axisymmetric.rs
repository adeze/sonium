//! Reduced axisymmetric CPU scattering with ring-source boundary elements.
//!
//! This module solves an axisymmetric rigid-sphere Neumann problem without
//! expanding a meridional trace into a three-dimensional surface mesh. Each
//! unknown multiplies an azimuthally integrated Helmholtz point-source ring;
//! the boundary equation is collocated on the generating semicircle. Sources
//! are placed on a checked interior concentric surface (the method of
//! fundamental solutions), so every kernel evaluation is regular.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::f64::consts::PI;
use thiserror::Error;

use crate::core::assembly::{CompressedTransmissionTbemSystem, TransmissionCompressionError};
use crate::core::types::PhysicsParams;

/// Failure from the reduced axisymmetric rigid-sphere path.
#[derive(Debug, Error)]
pub enum AxisymmetricBemError {
    /// Invalid geometry or discretization setting.
    #[error("invalid axisymmetric BEM parameter {name}: {value}")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// Exterior and interior media do not describe the same frequency.
    #[error("axisymmetric BEM frequencies differ: exterior {exterior}, interior {interior}")]
    FrequencyMismatch {
        /// Exterior frequency.
        exterior: f64,
        /// Interior frequency.
        interior: f64,
    },
    /// Exterior and interior media use incompatible harmonic signs.
    #[error("axisymmetric BEM harmonic factors differ: exterior {exterior}, interior {interior}")]
    HarmonicMismatch {
        /// Exterior harmonic sign.
        exterior: f64,
        /// Interior harmonic sign.
        interior: f64,
    },
    /// One of the supplied media failed physical validation.
    #[error("axisymmetric BEM invalid physics: {0}")]
    InvalidPhysics(String),
    /// A meridional profile and its normals have different lengths.
    #[error("axisymmetric BEM profile length {profile} does not match normal length {normals}")]
    ProfileLengthMismatch {
        /// Number of supplied profile points.
        profile: usize,
        /// Number of supplied normal vectors.
        normals: usize,
    },
    /// A supplied profile normal is not finite or non-zero.
    #[error("axisymmetric BEM profile normal {index} is invalid")]
    InvalidProfileNormal {
        /// Rejected normal index.
        index: usize,
    },
    /// A caller supplied a transmission RHS with the wrong length.
    #[error("axisymmetric BEM transmission RHS length {actual} does not match expected {expected}")]
    InvalidRhsLength {
        /// Required coupled-system RHS length.
        expected: usize,
        /// Supplied RHS length.
        actual: usize,
    },
    /// Direct factorization or solve failed.
    #[error("axisymmetric BEM direct solve failed: {0}")]
    DirectSolve(String),
    /// The independently recomputed collocation residual missed its bound.
    #[error("axisymmetric BEM relative residual {residual:e} exceeds {tolerance:e}")]
    Residual {
        /// Recomputed relative residual.
        residual: f64,
        /// Requested limit.
        tolerance: f64,
    },
    /// A requested field point is not in the exterior domain.
    #[error("axisymmetric BEM field point {index} is not exterior to the sphere")]
    NonExteriorPoint {
        /// Index in the supplied point slice.
        index: usize,
    },
    /// A requested interior field point was not strictly inside the sphere.
    #[error("axisymmetric BEM field point {index} not interior sphere")]
    NonInteriorPoint {
        /// Index in supplied point slice.
        index: usize,
    },
    /// A solve or field observable was non-finite.
    #[error("axisymmetric BEM produced non-finite {0}")]
    NonFinite(&'static str),
}

/// Reduced ring-source model of a sound-hard sphere in an axial plane wave.
#[derive(Debug, Clone)]
pub struct AxisymmetricRigidSphere {
    radius: f64,
    meridional_points: usize,
    source_radius_ratio: f64,
    azimuthal_samples: usize,
    physics: PhysicsParams,
}

impl AxisymmetricRigidSphere {
    /// Construct a checked reduced axisymmetric problem.
    pub fn try_new(
        radius: f64,
        meridional_points: usize,
        source_radius_ratio: f64,
        azimuthal_samples: usize,
        physics: PhysicsParams,
    ) -> Result<Self, AxisymmetricBemError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "radius",
                value: radius,
            });
        }
        if meridional_points < 4 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "meridional_points",
                value: meridional_points as f64,
            });
        }
        if !source_radius_ratio.is_finite()
            || source_radius_ratio <= 0.0
            || source_radius_ratio >= 1.0
        {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "source_radius_ratio",
                value: source_radius_ratio,
            });
        }
        if azimuthal_samples < 8 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "azimuthal_samples",
                value: azimuthal_samples as f64,
            });
        }
        physics
            .validate()
            .map_err(|error| AxisymmetricBemError::InvalidPhysics(error.to_string()))?;
        Ok(Self {
            radius,
            meridional_points,
            source_radius_ratio,
            azimuthal_samples,
            physics,
        })
    }

    /// Number of reduced meridional unknowns.
    pub fn num_dofs(&self) -> usize {
        self.meridional_points
    }

    /// Solve for an axial plane wave with the supplied complex amplitude.
    pub fn solve(
        &self,
        amplitude: Complex64,
        residual_tolerance: f64,
    ) -> Result<AxisymmetricRigidSphereSolution, AxisymmetricBemError> {
        if !complex_is_finite(amplitude) {
            return Err(AxisymmetricBemError::NonFinite("incident amplitude"));
        }
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }

        let count = self.meridional_points;
        let mut matrix = DenseMatrix::<Complex64>::zeros(count, count);
        let mut rhs = vec![Complex64::default(); count];
        for row in 0..count {
            let theta = meridional_angle(row, count);
            let target = sphere_point(self.radius, theta, 0.0);
            let normal = scale(target, 1.0 / self.radius);
            let incident = axial_incident(amplitude, target[2], &self.physics);
            rhs[row] = -Complex64::new(
                0.0,
                self.physics.harmonic_factor * self.physics.wave_number * normal[2],
            ) * incident;
            for column in 0..count {
                let source_theta = meridional_angle(column, count);
                matrix[[row, column]] = ring_normal_derivative(
                    target,
                    normal,
                    self.radius * self.source_radius_ratio,
                    source_theta,
                    self.azimuthal_samples,
                    &self.physics,
                );
            }
        }

        let coefficients = sonium_solvers::direct::lu_solve(&matrix, &rhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        if coefficients.iter().any(|&value| !complex_is_finite(value)) {
            return Err(AxisymmetricBemError::NonFinite("ring coefficient"));
        }
        let relative_residual = relative_residual(&matrix, &coefficients, &rhs);
        if !relative_residual.is_finite() {
            return Err(AxisymmetricBemError::NonFinite("relative residual"));
        }
        if relative_residual > residual_tolerance {
            return Err(AxisymmetricBemError::Residual {
                residual: relative_residual,
                tolerance: residual_tolerance,
            });
        }

        Ok(AxisymmetricRigidSphereSolution {
            problem: self.clone(),
            amplitude,
            coefficients,
            relative_residual,
        })
    }
}

/// Solved reduced axisymmetric rigid-sphere field.
#[derive(Debug, Clone)]
pub struct AxisymmetricRigidSphereSolution {
    problem: AxisymmetricRigidSphere,
    amplitude: Complex64,
    coefficients: Vec<Complex64>,
    relative_residual: f64,
}

impl AxisymmetricRigidSphereSolution {
    /// Independently recomputed dense collocation residual.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Reduced meridional unknown count.
    pub fn num_dofs(&self) -> usize {
        self.coefficients.len()
    }

    /// Evaluate total pressure at exterior `(radial, axial)` points.
    pub fn evaluate_total_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        points
            .iter()
            .enumerate()
            .map(|(index, &[radial, axial])| {
                if !radial.is_finite()
                    || !axial.is_finite()
                    || radial < 0.0
                    || radial.hypot(axial) <= self.problem.radius
                {
                    return Err(AxisymmetricBemError::NonExteriorPoint { index });
                }
                let target = [radial, 0.0, axial];
                let mut pressure = axial_incident(self.amplitude, axial, &self.problem.physics);
                for (column, &coefficient) in self.coefficients.iter().enumerate() {
                    pressure += coefficient
                        * ring_green(
                            target,
                            self.problem.radius * self.problem.source_radius_ratio,
                            meridional_angle(column, self.coefficients.len()),
                            self.problem.azimuthal_samples,
                            &self.problem.physics,
                        );
                }
                if !complex_is_finite(pressure) {
                    return Err(AxisymmetricBemError::NonFinite("field pressure"));
                }
                Ok(pressure)
            })
            .collect()
    }
}

/// Reduced axisymmetric CPU transmission problem for a spherical fluid inclusion.
///
/// The exterior scattered trace is represented by azimuthally integrated source
/// rings strictly inside the interface, while the interior trace uses a second
/// ring family strictly outside it. This keeps both kernels regular and avoids
/// expanding the axisymmetric problem into a three-dimensional mesh.
#[derive(Debug, Clone)]
pub struct AxisymmetricPenetrableSphere {
    radius: f64,
    meridional_points: usize,
    source_radius_ratio: f64,
    azimuthal_samples: usize,
    exterior: PhysicsParams,
    interior: PhysicsParams,
}

impl AxisymmetricPenetrableSphere {
    /// Build a checked two-medium axisymmetric sphere problem.
    pub fn try_new(
        radius: f64,
        meridional_points: usize,
        source_radius_ratio: f64,
        azimuthal_samples: usize,
        exterior: PhysicsParams,
        interior: PhysicsParams,
    ) -> Result<Self, AxisymmetricBemError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "radius",
                value: radius,
            });
        }
        if meridional_points < 4 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "meridional_points",
                value: meridional_points as f64,
            });
        }
        if !source_radius_ratio.is_finite()
            || source_radius_ratio <= 0.0
            || source_radius_ratio >= 1.0
        {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "source_radius_ratio",
                value: source_radius_ratio,
            });
        }
        if azimuthal_samples < 8 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "azimuthal_samples",
                value: azimuthal_samples as f64,
            });
        }
        exterior
            .validate()
            .map_err(|error| AxisymmetricBemError::InvalidPhysics(error.to_string()))?;
        interior
            .validate()
            .map_err(|error| AxisymmetricBemError::InvalidPhysics(error.to_string()))?;
        let frequency_scale = exterior
            .frequency
            .abs()
            .max(interior.frequency.abs())
            .max(1.0);
        if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
            return Err(AxisymmetricBemError::FrequencyMismatch {
                exterior: exterior.frequency,
                interior: interior.frequency,
            });
        }
        if exterior.harmonic_factor != interior.harmonic_factor {
            return Err(AxisymmetricBemError::HarmonicMismatch {
                exterior: exterior.harmonic_factor,
                interior: interior.harmonic_factor,
            });
        }
        Ok(Self {
            radius,
            meridional_points,
            source_radius_ratio,
            azimuthal_samples,
            exterior,
            interior,
        })
    }

    /// Number of meridional unknowns per medium.
    pub fn num_dofs(&self) -> usize {
        2 * self.meridional_points
    }

    /// Assemble the checked coupled transmission matrix and incident RHS.
    pub fn assemble_system(
        &self,
        amplitude: Complex64,
    ) -> Result<(DenseMatrix<Complex64>, Vec<Complex64>), AxisymmetricBemError> {
        if !complex_is_finite(amplitude) {
            return Err(AxisymmetricBemError::NonFinite("incident amplitude"));
        }
        let n = self.meridional_points;
        let mut matrix = DenseMatrix::<Complex64>::zeros(2 * n, 2 * n);
        let mut rhs = vec![Complex64::default(); 2 * n];
        let exterior_source_radius = self.radius * self.source_radius_ratio;
        let interior_source_radius = self.radius / self.source_radius_ratio;
        for row in 0..n {
            let theta = meridional_angle(row, n);
            let target = sphere_point(self.radius, theta, 0.0);
            let normal = scale(target, 1.0 / self.radius);
            let incident = axial_incident(amplitude, target[2], &self.exterior);
            let incident_derivative = Complex64::new(
                0.0,
                self.exterior.harmonic_factor * self.exterior.wave_number * normal[2],
            ) * incident;
            rhs[row] = -incident;
            rhs[n + row] = -incident_derivative / self.exterior.density;
            for column in 0..n {
                let source_theta = meridional_angle(column, n);
                matrix[[row, column]] = ring_green(
                    target,
                    exterior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.exterior,
                );
                matrix[[row, n + column]] = -ring_green(
                    target,
                    interior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.interior,
                );
                matrix[[n + row, column]] = ring_normal_derivative(
                    target,
                    normal,
                    exterior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.exterior,
                ) / self.exterior.density;
                matrix[[n + row, n + column]] = -ring_normal_derivative(
                    target,
                    normal,
                    interior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.interior,
                ) / self.interior.density;
            }
        }
        if matrix
            .as_slice()
            .iter()
            .any(|value| !complex_is_finite(*value))
            || rhs.iter().any(|value| !complex_is_finite(*value))
        {
            return Err(AxisymmetricBemError::NonFinite("transmission system"));
        }
        Ok((matrix, rhs))
    }

    /// Build a compressed sphere transmission operator directly from ring kernels.
    ///
    /// The callback preserves the dense transmission block ordering and signs
    /// while avoiding materialization of the full coupled matrix.
    pub fn compress(
        &self,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedTransmissionTbemSystem, TransmissionCompressionError> {
        let n = self.meridional_points;
        let exterior_source_radius = self.radius * self.source_radius_ratio;
        let interior_source_radius = self.radius / self.source_radius_ratio;
        let targets = (0..n)
            .map(|row| sphere_point(self.radius, meridional_angle(row, n), 0.0))
            .collect::<Vec<_>>();
        let normals = targets
            .iter()
            .copied()
            .map(|target| scale(target, 1.0 / self.radius))
            .collect::<Vec<_>>();
        let azimuthal_samples = self.azimuthal_samples;
        let exterior = &self.exterior;
        let interior = &self.interior;

        CompressedTransmissionTbemSystem::from_entry(
            n,
            2 * n,
            block_size,
            tolerance,
            max_rank,
            |row, column| {
                let row_profile = row % n;
                let source_theta = meridional_angle(column % n, n);
                let target = targets[row_profile];
                if row < n {
                    if column < n {
                        ring_green(
                            target,
                            exterior_source_radius,
                            source_theta,
                            azimuthal_samples,
                            exterior,
                        )
                    } else {
                        -ring_green(
                            target,
                            interior_source_radius,
                            source_theta,
                            azimuthal_samples,
                            interior,
                        )
                    }
                } else {
                    let normal = normals[row_profile];
                    if column < n {
                        ring_normal_derivative(
                            target,
                            normal,
                            exterior_source_radius,
                            source_theta,
                            azimuthal_samples,
                            exterior,
                        ) / exterior.density
                    } else {
                        -ring_normal_derivative(
                            target,
                            normal,
                            interior_source_radius,
                            source_theta,
                            azimuthal_samples,
                            interior,
                        ) / interior.density
                    }
                }
            },
        )
    }

    /// Solve the Hermitian-adjoint transmission system `Aᴴ y = rhs`.
    pub fn solve_adjoint(
        &self,
        rhs: &[Complex64],
        residual_tolerance: f64,
    ) -> Result<AxisymmetricTransmissionAdjointSolution, AxisymmetricBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }
        let expected = self.num_dofs();
        if rhs.len() != expected {
            return Err(AxisymmetricBemError::InvalidRhsLength {
                expected,
                actual: rhs.len(),
            });
        }
        if rhs.iter().any(|value| !complex_is_finite(*value)) {
            return Err(AxisymmetricBemError::NonFinite("adjoint RHS"));
        }
        let (matrix, _) = self.assemble_system(Complex64::default())?;
        let mut hermitian = DenseMatrix::<Complex64>::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                hermitian[[row, column]] = matrix[[column, row]].conj();
            }
        }
        let coefficients = sonium_solvers::direct::lu_solve(&hermitian, rhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        let residual = relative_residual(&hermitian, &coefficients, rhs);
        if !residual.is_finite() {
            return Err(AxisymmetricBemError::NonFinite("adjoint residual"));
        }
        if residual > residual_tolerance {
            return Err(AxisymmetricBemError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        Ok(AxisymmetricTransmissionAdjointSolution {
            coefficients,
            relative_residual: residual,
            problem: self.clone(),
        })
    }

    /// Solve several Hermitian-adjoint transmission systems with one
    /// assembled/factorized operator.
    ///
    /// The right-hand sides are independent columns of `Aᴴ Y = B`.  Input
    /// validation and the residual gate are applied to every column, so a
    /// malformed batch fails closed without returning a partial result.
    pub fn solve_adjoint_many(
        &self,
        rhs_columns: &[Vec<Complex64>],
        residual_tolerance: f64,
    ) -> Result<Vec<AxisymmetricTransmissionAdjointSolution>, AxisymmetricBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }
        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }
        let expected = self.num_dofs();
        for rhs in rhs_columns {
            if rhs.len() != expected {
                return Err(AxisymmetricBemError::InvalidRhsLength {
                    expected,
                    actual: rhs.len(),
                });
            }
            if rhs.iter().any(|value| !complex_is_finite(*value)) {
                return Err(AxisymmetricBemError::NonFinite("adjoint RHS"));
            }
        }

        let (matrix, _) = self.assemble_system(Complex64::default())?;
        let mut hermitian = DenseMatrix::<Complex64>::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                hermitian[[row, column]] = matrix[[column, row]].conj();
            }
        }

        let factorization = sonium_solvers::direct::lu_factorize(&hermitian)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        let nrhs = rhs_columns.len();
        let mut rhs_block = vec![Complex64::default(); expected * nrhs];
        for (column, rhs) in rhs_columns.iter().enumerate() {
            for (row, value) in rhs.iter().enumerate() {
                rhs_block[row * nrhs + column] = *value;
            }
        }
        let solution_block = factorization
            .solve_many(&rhs_block, nrhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;

        let mut solutions = Vec::with_capacity(nrhs);
        for column in 0..nrhs {
            let coefficients = (0..expected)
                .map(|row| solution_block[row * nrhs + column])
                .collect::<Vec<_>>();
            let residual = relative_residual(&hermitian, &coefficients, &rhs_columns[column]);
            if !residual.is_finite() {
                return Err(AxisymmetricBemError::NonFinite("adjoint residual"));
            }
            if residual > residual_tolerance {
                return Err(AxisymmetricBemError::Residual {
                    residual,
                    tolerance: residual_tolerance,
                });
            }
            solutions.push(AxisymmetricTransmissionAdjointSolution {
                coefficients,
                relative_residual: residual,
                problem: self.clone(),
            });
        }
        Ok(solutions)
    }

    /// Solve the two-medium problem for an axial plane wave.
    pub fn solve(
        &self,
        amplitude: Complex64,
        residual_tolerance: f64,
    ) -> Result<AxisymmetricPenetrableSphereSolution, AxisymmetricBemError> {
        if !complex_is_finite(amplitude) {
            return Err(AxisymmetricBemError::NonFinite("incident amplitude"));
        }
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }
        let n = self.meridional_points;
        let mut matrix = DenseMatrix::<Complex64>::zeros(2 * n, 2 * n);
        let mut rhs = vec![Complex64::default(); 2 * n];
        let exterior_source_radius = self.radius * self.source_radius_ratio;
        let interior_source_radius = self.radius / self.source_radius_ratio;

        for row in 0..n {
            let theta = meridional_angle(row, n);
            let target = sphere_point(self.radius, theta, 0.0);
            let normal = scale(target, 1.0 / self.radius);
            let incident = axial_incident(amplitude, target[2], &self.exterior);
            let incident_derivative = Complex64::new(
                0.0,
                self.exterior.harmonic_factor * self.exterior.wave_number * normal[2],
            ) * incident;
            rhs[row] = -incident;
            rhs[n + row] = -incident_derivative / self.exterior.density;

            for column in 0..n {
                let source_theta = meridional_angle(column, n);
                let exterior_single = ring_green(
                    target,
                    exterior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.exterior,
                );
                let interior_single = ring_green(
                    target,
                    interior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.interior,
                );
                matrix[[row, column]] = exterior_single;
                matrix[[row, n + column]] = -interior_single;
                matrix[[n + row, column]] = ring_normal_derivative(
                    target,
                    normal,
                    exterior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.exterior,
                ) / self.exterior.density;
                matrix[[n + row, n + column]] = -ring_normal_derivative(
                    target,
                    normal,
                    interior_source_radius,
                    source_theta,
                    self.azimuthal_samples,
                    &self.interior,
                ) / self.interior.density;
            }
        }

        let coefficients = sonium_solvers::direct::lu_solve(&matrix, &rhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        if coefficients.iter().any(|value| !complex_is_finite(*value)) {
            return Err(AxisymmetricBemError::NonFinite("transmission coefficient"));
        }
        let residual = relative_residual(&matrix, &coefficients, &rhs);
        if !residual.is_finite() {
            return Err(AxisymmetricBemError::NonFinite("transmission residual"));
        }
        if residual > residual_tolerance {
            return Err(AxisymmetricBemError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        Ok(AxisymmetricPenetrableSphereSolution {
            problem: self.clone(),
            amplitude,
            exterior_coefficients: coefficients[..n].to_vec(),
            interior_coefficients: coefficients[n..].to_vec(),
            relative_residual: residual,
        })
    }
}

/// A checked general-profile, two-medium axisymmetric transmission problem.
///
/// The profile is supplied as `(radial, axial)` meridional points together
/// with outward `(radial, axial)` normals.  Exterior and interior ring sources
/// are offset independently along those normals, so the operator does not
/// assume a sphere, concentric geometry, or a single global radius.  This is
/// still a collocation method-of-fundamental-solutions path; Galerkin and
/// accelerated profile operators remain separate capabilities.
#[derive(Debug, Clone)]
pub struct AxisymmetricPenetrableProfile {
    profile: Vec<[f64; 2]>,
    normals: Vec<[f64; 2]>,
    exterior_offset: f64,
    interior_offset: f64,
    azimuthal_samples: usize,
    exterior: PhysicsParams,
    interior: PhysicsParams,
}

impl AxisymmetricPenetrableProfile {
    /// Build a checked general-profile transmission problem.
    pub fn try_new(
        profile: Vec<[f64; 2]>,
        normals: Vec<[f64; 2]>,
        exterior_offset: f64,
        interior_offset: f64,
        azimuthal_samples: usize,
        exterior: PhysicsParams,
        interior: PhysicsParams,
    ) -> Result<Self, AxisymmetricBemError> {
        if profile.len() < 4 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "profile_points",
                value: profile.len() as f64,
            });
        }
        if profile.len() != normals.len() {
            return Err(AxisymmetricBemError::ProfileLengthMismatch {
                profile: profile.len(),
                normals: normals.len(),
            });
        }
        for (index, &[radial, axial]) in profile.iter().enumerate() {
            if !radial.is_finite() || !axial.is_finite() || radial < 0.0 {
                return Err(AxisymmetricBemError::InvalidParameter {
                    name: "profile_coordinate",
                    value: index as f64,
                });
            }
            let [normal_radial, normal_axial] = normals[index];
            let normal_norm = normal_radial.hypot(normal_axial);
            if !normal_radial.is_finite()
                || !normal_axial.is_finite()
                || !normal_norm.is_finite()
                || normal_norm <= f64::EPSILON
            {
                return Err(AxisymmetricBemError::InvalidProfileNormal { index });
            }
        }
        for (name, value) in [
            ("exterior_offset", exterior_offset),
            ("interior_offset", interior_offset),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(AxisymmetricBemError::InvalidParameter { name, value });
            }
        }
        if azimuthal_samples < 8 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "azimuthal_samples",
                value: azimuthal_samples as f64,
            });
        }
        validate_transmission_media(&exterior, &interior)?;

        let problem = Self {
            profile,
            normals: normals
                .into_iter()
                .map(|[radial, axial]| {
                    let scale = radial.hypot(axial);
                    [radial / scale, axial / scale]
                })
                .collect(),
            exterior_offset,
            interior_offset,
            azimuthal_samples,
            exterior,
            interior,
        };
        problem.validate_source_radii()?;
        problem.validate_source_separation()?;
        Ok(problem)
    }

    /// Number of meridional profile unknowns in the coupled system.
    pub fn num_dofs(&self) -> usize {
        2 * self.profile.len()
    }

    /// Return the profile points in `(radial, axial)` coordinates.
    pub fn profile(&self) -> &[[f64; 2]] {
        &self.profile
    }

    /// Assemble the checked collocation transmission matrix and plane-wave RHS.
    pub fn assemble_system(
        &self,
        amplitude: Complex64,
    ) -> Result<(DenseMatrix<Complex64>, Vec<Complex64>), AxisymmetricBemError> {
        if !complex_is_finite(amplitude) {
            return Err(AxisymmetricBemError::NonFinite("incident amplitude"));
        }
        let n = self.profile.len();
        let mut matrix = DenseMatrix::<Complex64>::zeros(2 * n, 2 * n);
        let mut rhs = vec![Complex64::default(); 2 * n];
        let exterior_sources = self.exterior_sources();
        let interior_sources = self.interior_sources();
        for row in 0..n {
            let target = profile_point(self.profile[row]);
            let normal = profile_normal(self.normals[row]);
            let incident = axial_incident(amplitude, target[2], &self.exterior);
            let incident_derivative = Complex64::new(
                0.0,
                self.exterior.harmonic_factor * self.exterior.wave_number * normal[2],
            ) * incident;
            rhs[row] = -incident;
            rhs[n + row] = -incident_derivative / self.exterior.density;
            for column in 0..n {
                matrix[[row, column]] = ring_green_profile(
                    target,
                    exterior_sources[column],
                    self.azimuthal_samples,
                    &self.exterior,
                );
                matrix[[row, n + column]] = -ring_green_profile(
                    target,
                    interior_sources[column],
                    self.azimuthal_samples,
                    &self.interior,
                );
                matrix[[n + row, column]] = ring_normal_derivative_profile(
                    target,
                    normal,
                    exterior_sources[column],
                    self.azimuthal_samples,
                    &self.exterior,
                ) / self.exterior.density;
                matrix[[n + row, n + column]] = -ring_normal_derivative_profile(
                    target,
                    normal,
                    interior_sources[column],
                    self.azimuthal_samples,
                    &self.interior,
                ) / self.interior.density;
            }
        }
        if matrix
            .as_slice()
            .iter()
            .any(|value| !complex_is_finite(*value))
            || rhs.iter().any(|value| !complex_is_finite(*value))
        {
            return Err(AxisymmetricBemError::NonFinite("transmission system"));
        }
        Ok((matrix, rhs))
    }

    /// Build a compressed transmission operator directly from ring-kernel entries.
    ///
    /// The callback path avoids materializing the dense profile matrix while
    /// preserving the same row/column ordering and signs as [`Self::assemble_system`].
    pub fn compress(
        &self,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedTransmissionTbemSystem, TransmissionCompressionError> {
        let n = self.profile.len();
        let exterior_sources = self.exterior_sources();
        let interior_sources = self.interior_sources();
        let targets = self
            .profile
            .iter()
            .copied()
            .map(profile_point)
            .collect::<Vec<_>>();
        let normals = self
            .normals
            .iter()
            .copied()
            .map(profile_normal)
            .collect::<Vec<_>>();
        let azimuthal_samples = self.azimuthal_samples;
        let exterior = &self.exterior;
        let interior = &self.interior;

        CompressedTransmissionTbemSystem::from_entry(
            n,
            2 * n,
            block_size,
            tolerance,
            max_rank,
            |row, column| {
                let row_profile = row % n;
                let column_profile = column % n;
                let target = targets[row_profile];
                if row < n {
                    if column < n {
                        ring_green_profile(
                            target,
                            exterior_sources[column_profile],
                            azimuthal_samples,
                            exterior,
                        )
                    } else {
                        -ring_green_profile(
                            target,
                            interior_sources[column_profile],
                            azimuthal_samples,
                            interior,
                        )
                    }
                } else {
                    let normal = normals[row_profile];
                    if column < n {
                        ring_normal_derivative_profile(
                            target,
                            normal,
                            exterior_sources[column_profile],
                            azimuthal_samples,
                            exterior,
                        ) / exterior.density
                    } else {
                        -ring_normal_derivative_profile(
                            target,
                            normal,
                            interior_sources[column_profile],
                            azimuthal_samples,
                            interior,
                        ) / interior.density
                    }
                }
            },
        )
    }

    /// Solve the general-profile transmission problem for an axial plane wave.
    pub fn solve(
        &self,
        amplitude: Complex64,
        residual_tolerance: f64,
    ) -> Result<AxisymmetricPenetrableProfileSolution, AxisymmetricBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }
        let (matrix, rhs) = self.assemble_system(amplitude)?;
        let coefficients = sonium_solvers::direct::lu_solve(&matrix, &rhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        if coefficients.iter().any(|value| !complex_is_finite(*value)) {
            return Err(AxisymmetricBemError::NonFinite("transmission coefficient"));
        }
        let residual = relative_residual(&matrix, &coefficients, &rhs);
        if !residual.is_finite() {
            return Err(AxisymmetricBemError::NonFinite("transmission residual"));
        }
        if residual > residual_tolerance {
            return Err(AxisymmetricBemError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        let n = self.profile.len();
        Ok(AxisymmetricPenetrableProfileSolution {
            problem: self.clone(),
            amplitude,
            exterior_coefficients: coefficients[..n].to_vec(),
            interior_coefficients: coefficients[n..].to_vec(),
            relative_residual: residual,
        })
    }

    /// Solve several forward profile systems while reusing one LU
    /// factorization. The profile operator is independent of the incident
    /// plane-wave amplitude, so the unit-amplitude RHS is scaled for every
    /// requested column. Each returned solution carries an independently
    /// recomputed residual against the original assembled matrix.
    pub fn solve_many(
        &self,
        amplitudes: &[Complex64],
        residual_tolerance: f64,
    ) -> Result<Vec<AxisymmetricPenetrableProfileSolution>, AxisymmetricBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }
        if amplitudes.is_empty() {
            return Ok(Vec::new());
        }
        if amplitudes
            .iter()
            .any(|amplitude| !complex_is_finite(*amplitude))
        {
            return Err(AxisymmetricBemError::NonFinite("incident amplitude"));
        }

        let unit_amplitude = Complex64::new(1.0, 0.0);
        let (matrix, unit_rhs) = self.assemble_system(unit_amplitude)?;
        let factorization = sonium_solvers::direct::lu_factorize(&matrix)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        let expected = self.num_dofs();
        let nrhs = amplitudes.len();
        let mut rhs_block = vec![Complex64::default(); expected * nrhs];
        for (column, amplitude) in amplitudes.iter().enumerate() {
            for (row, value) in unit_rhs.iter().enumerate() {
                rhs_block[row * nrhs + column] = *value * *amplitude;
            }
        }
        let solution_block = factorization
            .solve_many(&rhs_block, nrhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;

        let mut solutions = Vec::with_capacity(nrhs);
        let n = self.profile.len();
        for (column, amplitude) in amplitudes.iter().enumerate() {
            let coefficients = (0..expected)
                .map(|row| solution_block[row * nrhs + column])
                .collect::<Vec<_>>();
            if coefficients.iter().any(|value| !complex_is_finite(*value)) {
                return Err(AxisymmetricBemError::NonFinite("transmission coefficient"));
            }
            let rhs = unit_rhs
                .iter()
                .map(|value| *value * *amplitude)
                .collect::<Vec<_>>();
            let residual = relative_residual(&matrix, &coefficients, &rhs);
            if !residual.is_finite() {
                return Err(AxisymmetricBemError::NonFinite("transmission residual"));
            }
            if residual > residual_tolerance {
                return Err(AxisymmetricBemError::Residual {
                    residual,
                    tolerance: residual_tolerance,
                });
            }
            solutions.push(AxisymmetricPenetrableProfileSolution {
                problem: self.clone(),
                amplitude: *amplitude,
                exterior_coefficients: coefficients[..n].to_vec(),
                interior_coefficients: coefficients[n..].to_vec(),
                relative_residual: residual,
            });
        }
        Ok(solutions)
    }

    /// Solve the exact Hermitian-adjoint profile system `Aᴴ y = rhs`.
    pub fn solve_adjoint(
        &self,
        rhs: &[Complex64],
        residual_tolerance: f64,
    ) -> Result<AxisymmetricProfileTransmissionAdjointSolution, AxisymmetricBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }
        if rhs.len() != self.num_dofs() {
            return Err(AxisymmetricBemError::InvalidRhsLength {
                expected: self.num_dofs(),
                actual: rhs.len(),
            });
        }
        if rhs.iter().any(|value| !complex_is_finite(*value)) {
            return Err(AxisymmetricBemError::NonFinite("adjoint RHS"));
        }
        let (matrix, _) = self.assemble_system(Complex64::default())?;
        let mut hermitian = DenseMatrix::<Complex64>::zeros(self.num_dofs(), self.num_dofs());
        for row in 0..self.num_dofs() {
            for column in 0..self.num_dofs() {
                hermitian[[row, column]] = matrix[[column, row]].conj();
            }
        }
        let coefficients = sonium_solvers::direct::lu_solve(&hermitian, rhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        let residual = relative_residual(&hermitian, &coefficients, rhs);
        if !residual.is_finite() {
            return Err(AxisymmetricBemError::NonFinite("adjoint residual"));
        }
        if residual > residual_tolerance {
            return Err(AxisymmetricBemError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        Ok(AxisymmetricProfileTransmissionAdjointSolution {
            problem: self.clone(),
            coefficients,
            relative_residual: residual,
        })
    }

    /// Solve several Hermitian-adjoint profile systems while reusing one LU
    /// factorization. Every RHS is checked and every returned residual is
    /// independently recomputed before the batch is accepted.
    pub fn solve_adjoint_many(
        &self,
        rhs_columns: &[Vec<Complex64>],
        residual_tolerance: f64,
    ) -> Result<Vec<AxisymmetricProfileTransmissionAdjointSolution>, AxisymmetricBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(AxisymmetricBemError::InvalidParameter {
                name: "residual_tolerance",
                value: residual_tolerance,
            });
        }
        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }
        let expected = self.num_dofs();
        for rhs in rhs_columns {
            if rhs.len() != expected {
                return Err(AxisymmetricBemError::InvalidRhsLength {
                    expected,
                    actual: rhs.len(),
                });
            }
            if rhs.iter().any(|value| !complex_is_finite(*value)) {
                return Err(AxisymmetricBemError::NonFinite("adjoint RHS"));
            }
        }
        let (matrix, _) = self.assemble_system(Complex64::default())?;
        let mut hermitian = DenseMatrix::<Complex64>::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                hermitian[[row, column]] = matrix[[column, row]].conj();
            }
        }
        let factorization = sonium_solvers::direct::lu_factorize(&hermitian)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        let nrhs = rhs_columns.len();
        let mut rhs_block = vec![Complex64::default(); expected * nrhs];
        for (column, rhs) in rhs_columns.iter().enumerate() {
            for (row, value) in rhs.iter().enumerate() {
                rhs_block[row * nrhs + column] = *value;
            }
        }
        let solution_block = factorization
            .solve_many(&rhs_block, nrhs)
            .map_err(|error| AxisymmetricBemError::DirectSolve(error.to_string()))?;
        let mut solutions = Vec::with_capacity(nrhs);
        for column in 0..nrhs {
            let coefficients = (0..expected)
                .map(|row| solution_block[row * nrhs + column])
                .collect::<Vec<_>>();
            let residual = relative_residual(&hermitian, &coefficients, &rhs_columns[column]);
            if !residual.is_finite() {
                return Err(AxisymmetricBemError::NonFinite("adjoint residual"));
            }
            if residual > residual_tolerance {
                return Err(AxisymmetricBemError::Residual {
                    residual,
                    tolerance: residual_tolerance,
                });
            }
            solutions.push(AxisymmetricProfileTransmissionAdjointSolution {
                problem: self.clone(),
                coefficients,
                relative_residual: residual,
            });
        }
        Ok(solutions)
    }

    fn exterior_sources(&self) -> Vec<[f64; 2]> {
        self.profile
            .iter()
            .zip(&self.normals)
            .map(|(&[radial, axial], &[normal_radial, normal_axial])| {
                [
                    radial - self.exterior_offset * normal_radial,
                    axial - self.exterior_offset * normal_axial,
                ]
            })
            .collect()
    }

    fn interior_sources(&self) -> Vec<[f64; 2]> {
        self.profile
            .iter()
            .zip(&self.normals)
            .map(|(&[radial, axial], &[normal_radial, normal_axial])| {
                [
                    radial + self.interior_offset * normal_radial,
                    axial + self.interior_offset * normal_axial,
                ]
            })
            .collect()
    }

    fn validate_source_radii(&self) -> Result<(), AxisymmetricBemError> {
        for (index, (&[radial, _], &[normal_radial, _])) in
            self.profile.iter().zip(&self.normals).enumerate()
        {
            let exterior_radius = radial - self.exterior_offset * normal_radial;
            if !exterior_radius.is_finite() || exterior_radius <= 1.0e-12 {
                return Err(AxisymmetricBemError::InvalidParameter {
                    name: "exterior_source_radius",
                    value: index as f64,
                });
            }
            let interior_radius = radial + self.interior_offset * normal_radial;
            if !interior_radius.is_finite() || interior_radius <= 1.0e-12 {
                return Err(AxisymmetricBemError::InvalidParameter {
                    name: "interior_source_radius",
                    value: index as f64,
                });
            }
        }
        Ok(())
    }

    fn validate_source_separation(&self) -> Result<(), AxisymmetricBemError> {
        let exterior_sources = self.exterior_sources();
        let interior_sources = self.interior_sources();
        for (target_index, &target_profile) in self.profile.iter().enumerate() {
            let target = profile_point(target_profile);
            for &source in exterior_sources.iter().chain(&interior_sources) {
                let distance = norm(sub(target, profile_ring_point(source, 0.0)));
                if !distance.is_finite() || distance <= 1.0e-12 {
                    return Err(AxisymmetricBemError::InvalidParameter {
                        name: "source_separation",
                        value: target_index as f64,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Forward solution of [`AxisymmetricPenetrableProfile`].
#[derive(Debug, Clone)]
pub struct AxisymmetricPenetrableProfileSolution {
    problem: AxisymmetricPenetrableProfile,
    amplitude: Complex64,
    exterior_coefficients: Vec<Complex64>,
    interior_coefficients: Vec<Complex64>,
    relative_residual: f64,
}

impl AxisymmetricPenetrableProfileSolution {
    /// Return the independently certified relative residual.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Return exterior ring coefficients in profile order.
    pub fn exterior_coefficients(&self) -> &[Complex64] {
        &self.exterior_coefficients
    }

    /// Return interior ring coefficients in profile order.
    pub fn interior_coefficients(&self) -> &[Complex64] {
        &self.interior_coefficients
    }

    /// Evaluate the total exterior pressure at finite axisymmetric points.
    pub fn evaluate_exterior_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        let sources = self.problem.exterior_sources();
        points
            .iter()
            .enumerate()
            .map(|(index, &[radial, axial])| {
                if !radial.is_finite() || !axial.is_finite() || radial < 0.0 {
                    return Err(AxisymmetricBemError::NonFinite("exterior evaluation point"));
                }
                let target = [radial, 0.0, axial];
                let scattered = self
                    .exterior_coefficients
                    .iter()
                    .zip(&sources)
                    .map(|(&coefficient, &source)| {
                        coefficient
                            * ring_green_profile(
                                target,
                                source,
                                self.problem.azimuthal_samples,
                                &self.problem.exterior,
                            )
                    })
                    .sum::<Complex64>();
                let value =
                    axial_incident(self.amplitude, axial, &self.problem.exterior) + scattered;
                if !complex_is_finite(value) {
                    return Err(AxisymmetricBemError::NonFinite("exterior field"));
                }
                let _ = index;
                Ok(value)
            })
            .collect()
    }

    /// Evaluate the interior pressure at finite axisymmetric points.
    pub fn evaluate_interior_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        let sources = self.problem.interior_sources();
        points
            .iter()
            .map(|&[radial, axial]| {
                if !radial.is_finite() || !axial.is_finite() || radial < 0.0 {
                    return Err(AxisymmetricBemError::NonFinite("interior evaluation point"));
                }
                let target = [radial, 0.0, axial];
                let value = self
                    .interior_coefficients
                    .iter()
                    .zip(&sources)
                    .map(|(&coefficient, &source)| {
                        coefficient
                            * ring_green_profile(
                                target,
                                source,
                                self.problem.azimuthal_samples,
                                &self.problem.interior,
                            )
                    })
                    .sum::<Complex64>();
                if !complex_is_finite(value) {
                    return Err(AxisymmetricBemError::NonFinite("interior field"));
                }
                Ok(value)
            })
            .collect()
    }

    /// Independently recompute pressure continuity on every profile point.
    pub fn pressure_continuity_error(&self) -> Result<f64, AxisymmetricBemError> {
        let exterior_sources = self.problem.exterior_sources();
        let interior_sources = self.problem.interior_sources();
        let mut maximum = 0.0_f64;
        for (index, &profile) in self.problem.profile.iter().enumerate() {
            let target = profile_point(profile);
            let exterior = axial_incident(self.amplitude, target[2], &self.problem.exterior)
                + self
                    .exterior_coefficients
                    .iter()
                    .zip(&exterior_sources)
                    .map(|(&coefficient, &source)| {
                        coefficient
                            * ring_green_profile(
                                target,
                                source,
                                self.problem.azimuthal_samples,
                                &self.problem.exterior,
                            )
                    })
                    .sum::<Complex64>();
            let interior = self
                .interior_coefficients
                .iter()
                .zip(&interior_sources)
                .map(|(&coefficient, &source)| {
                    coefficient
                        * ring_green_profile(
                            target,
                            source,
                            self.problem.azimuthal_samples,
                            &self.problem.interior,
                        )
                })
                .sum::<Complex64>();
            let residual = exterior - interior;
            if !complex_is_finite(residual) {
                return Err(AxisymmetricBemError::NonFinite(
                    "profile pressure continuity",
                ));
            }
            let _ = index;
            maximum = maximum.max(residual.norm());
        }
        Ok(maximum)
    }

    /// Independently recompute density-weighted normal-velocity continuity.
    pub fn velocity_continuity_error(&self) -> Result<f64, AxisymmetricBemError> {
        let exterior_sources = self.problem.exterior_sources();
        let interior_sources = self.problem.interior_sources();
        let mut maximum = 0.0_f64;
        for (index, (&profile, &normal_profile)) in self
            .problem
            .profile
            .iter()
            .zip(&self.problem.normals)
            .enumerate()
        {
            let target = profile_point(profile);
            let normal = profile_normal(normal_profile);
            let incident = axial_incident(self.amplitude, target[2], &self.problem.exterior);
            let incident_derivative = Complex64::new(
                0.0,
                self.problem.exterior.harmonic_factor
                    * self.problem.exterior.wave_number
                    * normal[2],
            ) * incident;
            let exterior = incident_derivative
                + self
                    .exterior_coefficients
                    .iter()
                    .zip(&exterior_sources)
                    .map(|(&coefficient, &source)| {
                        coefficient
                            * ring_normal_derivative_profile(
                                target,
                                normal,
                                source,
                                self.problem.azimuthal_samples,
                                &self.problem.exterior,
                            )
                    })
                    .sum::<Complex64>();
            let interior = self
                .interior_coefficients
                .iter()
                .zip(&interior_sources)
                .map(|(&coefficient, &source)| {
                    coefficient
                        * ring_normal_derivative_profile(
                            target,
                            normal,
                            source,
                            self.problem.azimuthal_samples,
                            &self.problem.interior,
                        )
                })
                .sum::<Complex64>();
            let residual =
                exterior / self.problem.exterior.density - interior / self.problem.interior.density;
            if !complex_is_finite(residual) {
                return Err(AxisymmetricBemError::NonFinite(
                    "profile velocity continuity",
                ));
            }
            let _ = index;
            maximum = maximum.max(residual.norm());
        }
        Ok(maximum)
    }
}

/// Certified Hermitian-adjoint solution of a general-profile transmission system.
#[derive(Debug, Clone)]
pub struct AxisymmetricProfileTransmissionAdjointSolution {
    problem: AxisymmetricPenetrableProfile,
    coefficients: Vec<Complex64>,
    relative_residual: f64,
}

impl AxisymmetricProfileTransmissionAdjointSolution {
    /// Return coefficients in exterior-then-interior ordering.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    /// Return the independently certified relative residual.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Evaluate the adjoint exterior single-layer field at finite points.
    pub fn evaluate_exterior_adjoint_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        let n = self.problem.profile.len();
        let sources = self.problem.exterior_sources();
        points
            .iter()
            .map(|&[radial, axial]| {
                if !radial.is_finite() || !axial.is_finite() || radial < 0.0 {
                    return Err(AxisymmetricBemError::NonFinite("adjoint evaluation point"));
                }
                let target = [radial, 0.0, axial];
                let value = self.coefficients[..n]
                    .iter()
                    .zip(&sources)
                    .map(|(&coefficient, &source)| {
                        coefficient
                            * ring_green_profile(
                                target,
                                source,
                                self.problem.azimuthal_samples,
                                &self.problem.exterior,
                            )
                    })
                    .sum::<Complex64>();
                if !complex_is_finite(value) {
                    return Err(AxisymmetricBemError::NonFinite("adjoint field"));
                }
                Ok(value)
            })
            .collect()
    }

    /// Evaluate adjoint interior single-layer field at finite points.
    pub fn evaluate_interior_adjoint_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        let n = self.problem.profile.len();
        let sources = self.problem.interior_sources();
        points
            .iter()
            .map(|&[radial, axial]| {
                if !radial.is_finite() || !axial.is_finite() || radial < 0.0 {
                    return Err(AxisymmetricBemError::NonFinite(
                        "adjoint interior evaluation point",
                    ));
                }
                let target = [radial, 0.0, axial];
                let value = self.coefficients[n..]
                    .iter()
                    .zip(&sources)
                    .map(|(&coefficient, &source)| {
                        coefficient
                            * ring_green_profile(
                                target,
                                source,
                                self.problem.azimuthal_samples,
                                &self.problem.interior,
                            )
                    })
                    .sum::<Complex64>();
                if !complex_is_finite(value) {
                    return Err(AxisymmetricBemError::NonFinite("adjoint interior pressure"));
                }
                Ok(value)
            })
            .collect()
    }
}

/// Certified Hermitian-adjoint solution for [`AxisymmetricPenetrableSphere`].
#[derive(Debug, Clone)]
pub struct AxisymmetricTransmissionAdjointSolution {
    coefficients: Vec<Complex64>,
    relative_residual: f64,
    problem: AxisymmetricPenetrableSphere,
}

impl AxisymmetricTransmissionAdjointSolution {
    /// Return coefficients in exterior-then-interior operator ordering.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    /// Return the independently recomputed relative adjoint residual.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Evaluate the adjoint exterior single-layer pressure at exterior
    /// `(radial, axial)` points. The result is the solved adjoint trace
    /// representation, without an incident-field contribution.
    pub fn evaluate_exterior_adjoint_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        let n = self.problem.meridional_points;
        let source_radius = self.problem.radius * self.problem.source_radius_ratio;
        points
            .iter()
            .enumerate()
            .map(|(index, &[radial, axial])| {
                if !radial.is_finite()
                    || !axial.is_finite()
                    || radial < 0.0
                    || radial.hypot(axial) <= self.problem.radius
                {
                    return Err(AxisymmetricBemError::NonExteriorPoint { index });
                }
                let target = [radial, 0.0, axial];
                let pressure = self.coefficients[..n]
                    .iter()
                    .enumerate()
                    .map(|(column, &coefficient)| {
                        coefficient
                            * ring_green(
                                target,
                                source_radius,
                                meridional_angle(column, n),
                                self.problem.azimuthal_samples,
                                &self.problem.exterior,
                            )
                    })
                    .sum::<Complex64>();
                if !complex_is_finite(pressure) {
                    return Err(AxisymmetricBemError::NonFinite("adjoint exterior pressure"));
                }
                Ok(pressure)
            })
            .collect()
    }
}

/// Solution and field evaluator for [`AxisymmetricPenetrableSphere`].
#[derive(Debug, Clone)]
pub struct AxisymmetricPenetrableSphereSolution {
    problem: AxisymmetricPenetrableSphere,
    amplitude: Complex64,
    exterior_coefficients: Vec<Complex64>,
    interior_coefficients: Vec<Complex64>,
    relative_residual: f64,
}

impl AxisymmetricPenetrableSphereSolution {
    /// Number of unknowns across both media.
    pub fn num_dofs(&self) -> usize {
        self.exterior_coefficients.len() + self.interior_coefficients.len()
    }

    /// Independently recomputed coupled collocation residual.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Exterior scattered coefficients in meridional-ring order.
    pub fn exterior_coefficients(&self) -> &[Complex64] {
        &self.exterior_coefficients
    }

    /// Interior transmitted coefficients in meridional-ring order.
    pub fn interior_coefficients(&self) -> &[Complex64] {
        &self.interior_coefficients
    }

    /// Return the maximum pressure-continuity residual at meridional
    /// collocation points, independently re-evaluated from the solved rings.
    pub fn pressure_continuity_error(&self) -> Result<f64, AxisymmetricBemError> {
        let n = self.exterior_coefficients.len();
        let exterior_source_radius = self.problem.radius * self.problem.source_radius_ratio;
        let interior_source_radius = self.problem.radius / self.problem.source_radius_ratio;
        let mut maximum = 0.0_f64;
        for row in 0..n {
            let theta = meridional_angle(row, n);
            let target = sphere_point(self.problem.radius, theta, 0.0);
            let incident = axial_incident(self.amplitude, target[2], &self.problem.exterior);
            let exterior = incident
                + self
                    .exterior_coefficients
                    .iter()
                    .enumerate()
                    .map(|(column, &coefficient)| {
                        coefficient
                            * ring_green(
                                target,
                                exterior_source_radius,
                                meridional_angle(column, n),
                                self.problem.azimuthal_samples,
                                &self.problem.exterior,
                            )
                    })
                    .sum::<Complex64>();
            let interior = self
                .interior_coefficients
                .iter()
                .enumerate()
                .map(|(column, &coefficient)| {
                    coefficient
                        * ring_green(
                            target,
                            interior_source_radius,
                            meridional_angle(column, n),
                            self.problem.azimuthal_samples,
                            &self.problem.interior,
                        )
                })
                .sum::<Complex64>();
            let residual = exterior - interior;
            if !complex_is_finite(residual) {
                return Err(AxisymmetricBemError::NonFinite(
                    "pressure-continuity residual",
                ));
            }
            maximum = maximum.max(residual.norm());
        }
        Ok(maximum)
    }

    /// Return the maximum density-weighted normal-velocity continuity
    /// residual at meridional collocation points, independently re-evaluated
    /// from the solved rings.
    pub fn velocity_continuity_error(&self) -> Result<f64, AxisymmetricBemError> {
        let n = self.exterior_coefficients.len();
        let exterior_source_radius = self.problem.radius * self.problem.source_radius_ratio;
        let interior_source_radius = self.problem.radius / self.problem.source_radius_ratio;
        let mut maximum = 0.0_f64;
        for row in 0..n {
            let theta = meridional_angle(row, n);
            let target = sphere_point(self.problem.radius, theta, 0.0);
            let normal = scale(target, 1.0 / self.problem.radius);
            let incident = axial_incident(self.amplitude, target[2], &self.problem.exterior);
            let incident_derivative = Complex64::new(
                0.0,
                self.problem.exterior.harmonic_factor
                    * self.problem.exterior.wave_number
                    * normal[2],
            ) * incident;
            let exterior_derivative = self
                .exterior_coefficients
                .iter()
                .enumerate()
                .map(|(column, &coefficient)| {
                    coefficient
                        * ring_normal_derivative(
                            target,
                            normal,
                            exterior_source_radius,
                            meridional_angle(column, n),
                            self.problem.azimuthal_samples,
                            &self.problem.exterior,
                        )
                })
                .sum::<Complex64>();
            let interior_derivative = self
                .interior_coefficients
                .iter()
                .enumerate()
                .map(|(column, &coefficient)| {
                    coefficient
                        * ring_normal_derivative(
                            target,
                            normal,
                            interior_source_radius,
                            meridional_angle(column, n),
                            self.problem.azimuthal_samples,
                            &self.problem.interior,
                        )
                })
                .sum::<Complex64>();
            let exterior_velocity =
                (incident_derivative + exterior_derivative) / self.problem.exterior.density;
            let interior_velocity = interior_derivative / self.problem.interior.density;
            let residual = exterior_velocity - interior_velocity;
            if !complex_is_finite(residual) {
                return Err(AxisymmetricBemError::NonFinite(
                    "velocity-continuity residual",
                ));
            }
            maximum = maximum.max(residual.norm());
        }
        Ok(maximum)
    }

    /// Evaluate total exterior pressure at `(radial, axial)` points.
    pub fn evaluate_exterior_total_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        points
            .iter()
            .enumerate()
            .map(|(index, &[radial, axial])| {
                if !radial.is_finite()
                    || !axial.is_finite()
                    || radial < 0.0
                    || radial.hypot(axial) <= self.problem.radius
                {
                    return Err(AxisymmetricBemError::NonExteriorPoint { index });
                }
                let target = [radial, 0.0, axial];
                let mut pressure = axial_incident(self.amplitude, axial, &self.problem.exterior);
                for (column, &coefficient) in self.exterior_coefficients.iter().enumerate() {
                    pressure += coefficient
                        * ring_green(
                            target,
                            self.problem.radius * self.problem.source_radius_ratio,
                            meridional_angle(column, self.exterior_coefficients.len()),
                            self.problem.azimuthal_samples,
                            &self.problem.exterior,
                        );
                }
                if !complex_is_finite(pressure) {
                    return Err(AxisymmetricBemError::NonFinite("exterior field"));
                }
                Ok(pressure)
            })
            .collect()
    }

    /// Evaluate transmitted interior pressure at `(radial, axial)` points.
    pub fn evaluate_interior_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, AxisymmetricBemError> {
        points
            .iter()
            .enumerate()
            .map(|(index, &[radial, axial])| {
                if !radial.is_finite()
                    || !axial.is_finite()
                    || radial < 0.0
                    || radial.hypot(axial) >= self.problem.radius
                {
                    return Err(AxisymmetricBemError::NonInteriorPoint { index });
                }
                let target = [radial, 0.0, axial];
                let mut pressure = Complex64::default();
                for (column, &coefficient) in self.interior_coefficients.iter().enumerate() {
                    pressure += coefficient
                        * ring_green(
                            target,
                            self.problem.radius / self.problem.source_radius_ratio,
                            meridional_angle(column, self.interior_coefficients.len()),
                            self.problem.azimuthal_samples,
                            &self.problem.interior,
                        );
                }
                if !complex_is_finite(pressure) {
                    return Err(AxisymmetricBemError::NonFinite("interior field"));
                }
                Ok(pressure)
            })
            .collect()
    }
}

fn meridional_angle(index: usize, count: usize) -> f64 {
    PI * (index as f64 + 0.5) / count as f64
}

fn sphere_point(radius: f64, theta: f64, phi: f64) -> [f64; 3] {
    [
        radius * theta.sin() * phi.cos(),
        radius * theta.sin() * phi.sin(),
        radius * theta.cos(),
    ]
}

fn axial_incident(amplitude: Complex64, axial: f64, physics: &PhysicsParams) -> Complex64 {
    amplitude * Complex64::from_polar(1.0, physics.harmonic_factor * physics.wave_number * axial)
}

fn validate_transmission_media(
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
) -> Result<(), AxisymmetricBemError> {
    exterior
        .validate()
        .map_err(|error| AxisymmetricBemError::InvalidPhysics(error.to_string()))?;
    interior
        .validate()
        .map_err(|error| AxisymmetricBemError::InvalidPhysics(error.to_string()))?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(AxisymmetricBemError::FrequencyMismatch {
            exterior: exterior.frequency,
            interior: interior.frequency,
        });
    }
    if (exterior.harmonic_factor - interior.harmonic_factor).abs() > 1.0e-12 {
        return Err(AxisymmetricBemError::HarmonicMismatch {
            exterior: exterior.harmonic_factor,
            interior: interior.harmonic_factor,
        });
    }
    Ok(())
}

fn profile_point([radial, axial]: [f64; 2]) -> [f64; 3] {
    [radial, 0.0, axial]
}

fn profile_normal([radial, axial]: [f64; 2]) -> [f64; 3] {
    [radial, 0.0, axial]
}

fn profile_ring_point([radial, axial]: [f64; 2], phi: f64) -> [f64; 3] {
    [radial * phi.cos(), radial * phi.sin(), axial]
}

fn ring_green_profile(
    target: [f64; 3],
    source: [f64; 2],
    azimuthal_samples: usize,
    physics: &PhysicsParams,
) -> Complex64 {
    let mut sum = Complex64::default();
    for sample in 0..azimuthal_samples {
        let phi = 2.0 * PI * sample as f64 / azimuthal_samples as f64;
        let source_point = profile_ring_point(source, phi);
        let distance = norm(sub(target, source_point));
        sum += Complex64::from_polar(
            1.0 / (4.0 * PI * distance),
            physics.harmonic_factor * physics.wave_number * distance,
        );
    }
    sum / azimuthal_samples as f64
}

fn ring_normal_derivative_profile(
    target: [f64; 3],
    target_normal: [f64; 3],
    source: [f64; 2],
    azimuthal_samples: usize,
    physics: &PhysicsParams,
) -> Complex64 {
    let mut sum = Complex64::default();
    for sample in 0..azimuthal_samples {
        let phi = 2.0 * PI * sample as f64 / azimuthal_samples as f64;
        let source_point = profile_ring_point(source, phi);
        let displacement = sub(target, source_point);
        let distance = norm(displacement);
        let phase = Complex64::from_polar(
            1.0,
            physics.harmonic_factor * physics.wave_number * distance,
        );
        let radial_factor = Complex64::new(
            -1.0,
            physics.harmonic_factor * physics.wave_number * distance,
        ) / (4.0 * PI * distance.powi(3));
        sum += phase * radial_factor * dot(displacement, target_normal);
    }
    sum / azimuthal_samples as f64
}

fn ring_green(
    target: [f64; 3],
    source_radius: f64,
    source_theta: f64,
    azimuthal_samples: usize,
    physics: &PhysicsParams,
) -> Complex64 {
    let mut sum = Complex64::default();
    for sample in 0..azimuthal_samples {
        let phi = 2.0 * PI * sample as f64 / azimuthal_samples as f64;
        let source = sphere_point(source_radius, source_theta, phi);
        let distance = norm(sub(target, source));
        sum += Complex64::from_polar(
            1.0 / (4.0 * PI * distance),
            physics.harmonic_factor * physics.wave_number * distance,
        );
    }
    sum / azimuthal_samples as f64
}

fn ring_normal_derivative(
    target: [f64; 3],
    target_normal: [f64; 3],
    source_radius: f64,
    source_theta: f64,
    azimuthal_samples: usize,
    physics: &PhysicsParams,
) -> Complex64 {
    let mut sum = Complex64::default();
    for sample in 0..azimuthal_samples {
        let phi = 2.0 * PI * sample as f64 / azimuthal_samples as f64;
        let source = sphere_point(source_radius, source_theta, phi);
        let displacement = sub(target, source);
        let distance = norm(displacement);
        let phase = Complex64::from_polar(
            1.0,
            physics.harmonic_factor * physics.wave_number * distance,
        );
        let radial_factor = Complex64::new(
            -1.0,
            physics.harmonic_factor * physics.wave_number * distance,
        ) / (4.0 * PI * distance.powi(3));
        sum += phase * radial_factor * dot(displacement, target_normal);
    }
    sum / azimuthal_samples as f64
}

fn relative_residual(
    matrix: &DenseMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let mut residual_norm_squared = 0.0;
    let mut rhs_norm_squared = 0.0;
    for row in 0..rhs.len() {
        let mut value = -rhs[row];
        for column in 0..solution.len() {
            value += matrix[[row, column]] * solution[column];
        }
        residual_norm_squared += value.norm_sqr();
        rhs_norm_squared += rhs[row].norm_sqr();
    }
    residual_norm_squared.sqrt() / rhs_norm_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_constructor_rejects_invalid_reduction_settings() {
        let physics = PhysicsParams::try_new(100.0, 343.0, 1.21, false).unwrap();
        assert!(AxisymmetricRigidSphere::try_new(0.0, 16, 0.7, 32, physics.clone()).is_err());
        assert!(AxisymmetricRigidSphere::try_new(0.1, 3, 0.7, 32, physics.clone()).is_err());
        assert!(AxisymmetricRigidSphere::try_new(0.1, 16, 1.0, 32, physics.clone()).is_err());
        assert!(AxisymmetricRigidSphere::try_new(0.1, 16, 0.7, 7, physics).is_err());
    }

    #[test]
    fn checked_constructor_rejects_invalid_physics() {
        let mut physics = PhysicsParams::try_new(100.0, 343.0, 1.21, false).unwrap();
        physics.frequency = f64::NAN;

        let result = AxisymmetricRigidSphere::try_new(0.1, 16, 0.7, 32, physics);

        assert!(matches!(
            result,
            Err(AxisymmetricBemError::InvalidPhysics(_))
        ));
    }
}
