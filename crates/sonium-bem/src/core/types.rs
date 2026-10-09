//! Core type definitions for BEM solver
//!
//! This module defines the fundamental data structures used throughout the BEM implementation,
//! matching the C++ structures from NC_TypeDefinition.h while using idiomatic Rust types.
//!
//! # Design note: separate from sonium-mesher
//!
//! BEM mesh types (`Mesh`, `Element`, `ElementType`) are intentionally separate
//! from `sonium-mesher` because the two have fundamentally different needs:
//! - **Node storage**: BEM uses the crate-local array surface (cache-coherent row-major
//!   for tight integration kernels). Mesher uses `Vec<Point>` (per-node mutation
//!   during refinement).
//! - **Element data**: BEM elements carry precomputed normals, area, center, and
//!   DOF addresses (used billions of times in integration). Mesher elements are
//!   lightweight (geometry computed on-the-fly from quadrature).
//! - **Element types**: BEM is surface-only (Tri3/Quad4). Mesher includes volume
//!   elements (Tetrahedron, Hexahedron).
//! - **Boundary conditions**: BEM BCs carry complex-valued admittance data for
//!   acoustics. Mesher BCs are mathematical (Dirichlet/Neumann/Robin).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use sonium_solvers::dense::DenseMatrix;

use super::conventions::{ACOUSTIC_CONVENTION, BemConvention, BemDomain};

// ============================================================================
// Physical Constants and Parameters
// ============================================================================

/// Physical parameters for the acoustic problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    /// Speed of sound in the medium (m/s)
    pub speed_of_sound: f64,
    /// Density of the medium (kg/m³)
    pub density: f64,
    /// Current frequency (Hz)
    pub frequency: f64,
    /// Wave number k = ω/c = 2πf/c
    pub wave_number: f64,
    /// Angular frequency ω = 2πf
    pub omega: f64,
    /// Wave length λ = c/f
    pub wave_length: f64,
    /// Spatial Green-function sign for the shared `e^{-iωt}` convention.
    pub harmonic_factor: f64,
    /// Factor for sound pressure: ρ * ω * harmonic_factor
    pub pressure_factor: f64,
    /// Tau: +1 for external problem, -1 for internal problem
    pub tau: f64,
}

/// Failure while constructing or validating acoustic physical parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsParamsError {
    /// A primary medium/frequency quantity was not finite and positive.
    InvalidInput {
        /// Name of the invalid quantity.
        field: &'static str,
        /// Supplied value.
        value: f64,
    },
    /// The convention-derived Green-function sign was not ±1.
    InvalidHarmonicFactor {
        /// Supplied sign.
        value: f64,
    },
    /// The interior/exterior orientation marker was not ±1.
    InvalidTau {
        /// Supplied orientation marker.
        value: f64,
    },
    /// The exchange convention cannot be consumed by BEM.
    InvalidConvention {
        /// Human-readable reason from the convention adapter.
        reason: String,
    },
    /// A serialized or manually assembled derived quantity is inconsistent.
    InconsistentDerived {
        /// Name of the derived quantity.
        field: &'static str,
        /// Value implied by the primary inputs.
        expected: f64,
        /// Value present in the struct.
        actual: f64,
    },
}

impl std::fmt::Display for PhysicsParamsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { field, value } => {
                write!(f, "{field} must be finite and positive (got {value})")
            }
            Self::InvalidHarmonicFactor { value } => {
                write!(f, "harmonic_factor must be +1 or -1 (got {value})")
            }
            Self::InvalidTau { value } => write!(f, "tau must be +1 or -1 (got {value})"),
            Self::InvalidConvention { reason } => {
                write!(f, "invalid acoustic convention: {reason}")
            }
            Self::InconsistentDerived {
                field,
                expected,
                actual,
            } => write!(
                f,
                "{field} is inconsistent with the primary parameters (expected {expected}, got {actual})"
            ),
        }
    }
}

impl std::error::Error for PhysicsParamsError {}

impl PhysicsParams {
    /// Create new physics parameters from frequency and medium properties
    pub fn new(frequency: f64, speed_of_sound: f64, density: f64, is_internal: bool) -> Self {
        Self::new_with_convention(
            frequency,
            speed_of_sound,
            density,
            is_internal,
            ACOUSTIC_CONVENTION,
        )
    }

    /// Construct validated physical parameters from scalar inputs.
    ///
    /// [`Self::new`] remains available for compatibility with trusted
    /// numerical kernels. Exchange/configuration boundaries should prefer
    /// this fallible constructor so non-physical media and non-finite values
    /// are rejected before any wavenumber-dependent allocation or assembly.
    pub fn try_new(
        frequency: f64,
        speed_of_sound: f64,
        density: f64,
        is_internal: bool,
    ) -> Result<Self, PhysicsParamsError> {
        let params = Self::new(frequency, speed_of_sound, density, is_internal);
        params.validate()?;
        Ok(params)
    }

    /// Construct physical parameters from validated XEM quantities.
    ///
    /// This is the preferred exchange-facing boundary: units are represented
    /// by the argument types, so a length, pressure, or phase scalar cannot be
    /// accidentally passed as a frequency or material density.
    pub fn try_new_typed(
        frequency: sonium_xem::FrequencyHz,
        speed_of_sound: sonium_xem::SpeedOfSoundMPerS,
        density: sonium_xem::DensityKgM3,
        is_internal: bool,
    ) -> Result<Self, PhysicsParamsError> {
        Self::try_new(
            frequency.get(),
            speed_of_sound.get(),
            density.get(),
            is_internal,
        )
    }

    /// Construct from one convention-carrying XEM acoustic medium.
    pub fn try_new_medium(
        medium: sonium_xem::AcousticMedium,
        is_internal: bool,
    ) -> Result<Self, PhysicsParamsError> {
        let convention = BemConvention::from_exchange(&medium.conventions).map_err(|error| {
            PhysicsParamsError::InvalidConvention {
                reason: error.to_string(),
            }
        })?;
        Self::try_new_with_convention(
            medium.frequency.get(),
            medium.speed_of_sound.get(),
            medium.density.get(),
            is_internal,
            convention,
        )
    }

    /// Create physics parameters from an explicit exchange convention.
    ///
    /// Keeping this constructor beside [`Self::new`] prevents a caller that
    /// has already validated an XEM convention from silently falling back to
    /// the process-wide default phasor sign.
    pub fn new_with_convention(
        frequency: f64,
        speed_of_sound: f64,
        density: f64,
        is_internal: bool,
        convention: BemConvention,
    ) -> Self {
        use std::f64::consts::PI;
        let omega = 2.0 * PI * frequency;
        let wave_number = omega / speed_of_sound;
        let wave_length = speed_of_sound / frequency;
        // Keep the Green-function and pressure signs derived from the
        // validated convention so local assembly code cannot silently choose
        // a different phasor sign.
        let harmonic_factor = f64::from(convention.spatial_sign);
        let tau = BemDomain::from_internal(is_internal).tau();

        Self {
            speed_of_sound,
            density,
            frequency,
            wave_number,
            omega,
            wave_length,
            harmonic_factor,
            pressure_factor: density * omega * harmonic_factor,
            tau,
        }
    }

    /// Construct validated physical parameters using an explicit convention.
    pub fn try_new_with_convention(
        frequency: f64,
        speed_of_sound: f64,
        density: f64,
        is_internal: bool,
        convention: BemConvention,
    ) -> Result<Self, PhysicsParamsError> {
        let params =
            Self::new_with_convention(frequency, speed_of_sound, density, is_internal, convention);
        params.validate()?;
        Ok(params)
    }

    /// Validate both primary inputs and all convention-derived fields.
    ///
    /// The fields are public for serialization and compatibility, so callers
    /// can receive a malformed `PhysicsParams` by deserializing or manually
    /// assembling it. High-level solve/admission APIs call this method before
    /// entering an integration kernel.
    pub fn validate(&self) -> Result<(), PhysicsParamsError> {
        for (field, value) in [
            ("frequency", self.frequency),
            ("speed_of_sound", self.speed_of_sound),
            ("density", self.density),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PhysicsParamsError::InvalidInput { field, value });
            }
        }
        if !self.harmonic_factor.is_finite()
            || (self.harmonic_factor - 1.0).abs() > f64::EPSILON
                && (self.harmonic_factor + 1.0).abs() > f64::EPSILON
        {
            return Err(PhysicsParamsError::InvalidHarmonicFactor {
                value: self.harmonic_factor,
            });
        }
        if !self.tau.is_finite()
            || (self.tau - 1.0).abs() > f64::EPSILON && (self.tau + 1.0).abs() > f64::EPSILON
        {
            return Err(PhysicsParamsError::InvalidTau { value: self.tau });
        }

        let omega = 2.0 * std::f64::consts::PI * self.frequency;
        let derived = [
            ("omega", self.omega, omega),
            ("wave_number", self.wave_number, omega / self.speed_of_sound),
            (
                "wave_length",
                self.wave_length,
                self.speed_of_sound / self.frequency,
            ),
            (
                "pressure_factor",
                self.pressure_factor,
                self.density * omega * self.harmonic_factor,
            ),
        ];
        for (field, actual, expected) in derived {
            if !actual.is_finite() {
                return Err(PhysicsParamsError::InconsistentDerived {
                    field,
                    expected,
                    actual,
                });
            }
            let scale = expected.abs().max(actual.abs()).max(1.0);
            if (actual - expected).abs() > 1.0e-12 * scale {
                return Err(PhysicsParamsError::InconsistentDerived {
                    field,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Return a copy using an explicit acoustic convention.
    pub fn with_convention(mut self, convention: BemConvention) -> Self {
        self.harmonic_factor = f64::from(convention.spatial_sign);
        self.pressure_factor = self.density * self.omega * self.harmonic_factor;
        self
    }

    /// Burton-Miller coupling constant: β = i/k (for external problems)
    ///
    /// This is the traditional formulation. For better conditioning at low
    /// frequencies, use `burton_miller_beta_bounded` instead.
    pub fn burton_miller_beta(&self) -> Complex64 {
        if self.tau > 0.0 {
            Complex64::new(0.0, self.harmonic_factor / self.wave_number)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Derivative of the traditional Burton--Miller coupling with respect to
    /// wavenumber while speed, density, and geometry remain fixed.
    pub fn burton_miller_beta_wavenumber_derivative(&self) -> Complex64 {
        if self.tau > 0.0 {
            Complex64::new(
                0.0,
                -self.harmonic_factor / (self.wave_number * self.wave_number),
            )
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Derivative `dk/df` for a frequency perturbation at fixed sound speed.
    pub fn wave_number_frequency_derivative(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.speed_of_sound
    }

    /// Derivative of the pressure/velocity conversion factor with respect to
    /// wavenumber at fixed sound speed and density.
    pub fn pressure_factor_wavenumber_derivative(&self) -> f64 {
        self.density * self.speed_of_sound * self.harmonic_factor
    }

    /// Bounded Burton-Miller coupling: β = i / (k + k_ref)
    ///
    /// This formulation avoids the 1/k divergence at low frequencies.
    /// A typical choice for k_ref is 1/(average element size), which ensures
    /// the hypersingular contribution is properly scaled relative to the
    /// geometry-dependent edge integral term.
    ///
    /// # Arguments
    /// * `k_ref` - Reference wavenumber (should be ~1/element_size for best conditioning)
    pub fn burton_miller_beta_bounded(&self, k_ref: f64) -> Complex64 {
        if self.tau > 0.0 {
            Complex64::new(0.0, self.harmonic_factor / (self.wave_number + k_ref))
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Floored Burton-Miller beta to prevent ill-conditioning at high frequencies
    ///
    /// At high k, the traditional β = i/k becomes very small, making the matrix
    /// diagonal shrink faster than the RHS grows. This leads to O(k²) solution error.
    ///
    /// This method uses β = i * max(1/k, η_min) where η_min is chosen to keep
    /// |β * E_self| bounded away from zero.
    ///
    /// # Arguments
    /// * `edge_e_magnitude` - Approximate magnitude of the edge E integral (~50-100 typical)
    /// * `min_beta_e` - Minimum acceptable |β*E| (default: 5.0 for good conditioning)
    pub fn burton_miller_beta_floored(&self, edge_e_magnitude: f64, min_beta_e: f64) -> Complex64 {
        if self.tau > 0.0 {
            // Traditional: η = 1/k
            let eta_traditional = 1.0 / self.wave_number;

            // Minimum η to keep |β*E| ≥ min_beta_e
            let eta_min = min_beta_e / edge_e_magnitude;

            // Use the larger of the two
            let eta = eta_traditional.max(eta_min);

            Complex64::new(0.0, self.harmonic_factor * eta)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Optimal Burton-Miller beta based on mesh size (bounded version)
    ///
    /// Uses β = i/(k + k_ref) to bound β at low frequencies.
    /// Note: This may not help at high frequencies where the problem is different.
    ///
    /// # Arguments
    /// * `element_size` - Characteristic element size (edge length or sqrt(area))
    pub fn burton_miller_beta_optimal(&self, element_size: f64) -> Complex64 {
        if self.tau > 0.0 {
            let k_ref = 1.0 / element_size;
            Complex64::new(0.0, self.harmonic_factor / (self.wave_number + k_ref))
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Scaled Burton-Miller beta for better numerical conditioning
    ///
    /// Uses β = scale × i/k where scale > 1 improves diagonal dominance.
    /// Testing shows scale = 2.0 gives significantly better accuracy than the
    /// standard β = i/k, especially at low to medium frequencies (ka < 2).
    ///
    /// The improvement comes from increased diagonal dominance which improves
    /// matrix conditioning, outweighing the slight increase in row sum bias.
    ///
    /// # Arguments
    /// * `scale` - Scaling factor (recommended: 2.0)
    pub fn burton_miller_beta_scaled(&self, scale: f64) -> Complex64 {
        if self.tau > 0.0 {
            Complex64::new(0.0, self.harmonic_factor * scale / self.wave_number)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Adaptive Burton-Miller beta that varies with frequency
    ///
    /// Select the Mie-validated non-zero Burton-Miller scale.
    ///
    /// The validated constant-panel range uses `β = 0.2 i/k`.  A fixed
    /// exterior formulation sign and coupling scale preserve the same
    /// boundary-integral equation throughout a frequency sweep.
    ///
    /// # Arguments
    /// * `_radius` - Retained for API compatibility; the validated scale is
    ///   frequency-independent.
    ///
    /// # Returns
    /// Tuple of (beta, scale) where scale is the selected β scale factor
    pub fn burton_miller_beta_adaptive(&self, _radius: f64) -> (Complex64, f64) {
        if self.tau <= 0.0 {
            return (Complex64::new(0.0, 0.0), 1.0);
        }

        // A non-zero Burton-Miller term removes the irregular frequencies.
        // Complex Mie sweeps on shape-regular constant-panel sphere meshes
        // show that 0.2 i/k gives the lowest boundary-trace error throughout
        // the validated ka range; the former 4-16 i/k schedule amplified the
        // hypersingular discretisation error and eventually reversed mesh
        // convergence.
        let scale = 0.2;

        let beta = Complex64::new(0.0, self.harmonic_factor * scale / self.wave_number);
        (beta, scale)
    }

    /// Get optimal β scale for a given ka value
    ///
    /// Returns just the scale factor without computing β.
    /// Useful when you need to know the scale for matrix assembly.
    ///
    /// This matches the Mie-validated value used in
    /// `burton_miller_beta_adaptive` over the supported frequency range.
    pub fn optimal_beta_scale(_ka: f64) -> f64 {
        0.2
    }

    /// Return the canonical exterior-formulation sign.
    ///
    /// The sign is fixed by the boundary-integral convention, not a tuning
    /// parameter.  Flipping it at `ka = 0.5` changes the solved exterior
    /// equation and produces a 4-12% complex surface-field error even as the
    /// mesh is refined.
    pub fn optimal_formulation_sign(_ka: f64) -> f64 {
        1.0
    }

    /// Coupling constant γ (typically 1.0)
    pub fn gamma(&self) -> f64 {
        1.0
    }

    /// Return the convention used to derive these physical parameters.
    pub const fn convention() -> BemConvention {
        ACOUSTIC_CONVENTION
    }
}

// ============================================================================
// Mesh Structures
// ============================================================================

/// Basis function order for element approximation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasisOrder {
    /// Constant elements: 1 DOF per element, collocation at centroid
    Constant,
    /// Discontinuous linear elements: N_nodes DOFs per element (3 for Tri3, 4 for Quad4),
    /// collocation at internal Gauss points
    Linear,
    /// Discontinuous quadratic panel basis: 6 DOFs for Tri3 geometry and 9 DOFs for
    /// Quad4 geometry. Geometry remains the existing affine/bilinear panel; the
    /// approximation order applies to the unknown on each panel independently.
    Quadratic,
}

/// Continuity contract for higher-order surface unknowns.
///
/// [`Discontinuous`](Self::Discontinuous) is the historical panel-local
/// layout and remains the default. [`Continuous`](Self::Continuous) is
/// implemented for conforming linear and quadratic Tri3/Quad4 panels.
/// Quadratic layouts use the reusable
/// [`crate::core::mesh::ContinuousQuadraticDofMap`], while linear layouts
/// reuse mesh vertex numbering directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BasisContinuity {
    /// Every panel owns its local basis coefficients.
    #[default]
    Discontinuous,
    /// Shared vertex coefficients on conforming linear panels, or shared
    /// vertex/edge-midpoint coefficients on quadratic Tri3/Quad4 panels;
    /// Quad4 center coefficients remain panel-local.
    Continuous,
}

/// Element type (triangular or quadrilateral)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementType {
    /// 3-node triangle
    Tri3,
    /// 4-node quadrilateral
    Quad4,
}

impl ElementType {
    /// Number of nodes for this element type
    pub fn num_nodes(&self) -> usize {
        match self {
            ElementType::Tri3 => 3,
            ElementType::Quad4 => 4,
        }
    }

    /// Number of DOFs for this element type at the given basis order
    pub fn num_dofs_for_basis(&self, basis: BasisOrder) -> usize {
        match basis {
            BasisOrder::Constant => 1,
            BasisOrder::Linear => self.num_nodes(),
            BasisOrder::Quadratic => match self {
                ElementType::Tri3 => 6,
                ElementType::Quad4 => 9,
            },
        }
    }
}

/// Element property (from listElementProperty in C++)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementProperty {
    /// Surface element (boundary)
    Surface = 0,
    /// Middle face element
    MidFace = 1,
    /// Evaluation element (for post-processing)
    Evaluation = 2,
}

impl ElementProperty {
    /// Check if this is an evaluation element
    pub fn is_evaluation(&self) -> bool {
        matches!(self, ElementProperty::Evaluation)
    }

    /// Check if this is a surface element
    pub fn is_surface(&self) -> bool {
        matches!(self, ElementProperty::Surface)
    }
}

/// Boundary condition type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryCondition {
    /// Velocity prescribed (Neumann BC)
    Velocity(Vec<Complex64>),
    /// Pressure prescribed (Dirichlet BC)
    Pressure(Vec<Complex64>),
    /// Velocity with surface admittance
    VelocityWithAdmittance {
        /// Velocity values at each node
        velocity: Vec<Complex64>,
        /// Surface admittance value
        admittance: Complex64,
    },
    /// Local transfer admittance.
    ///
    /// In the current single-surface BEM assembly this is treated as a local
    /// admittance closure with zero prescribed velocity. A future coupled-panel
    /// assembly can use the same variant for cross-surface transfer DOFs.
    TransferAdmittance {
        /// Transfer admittance value
        admittance: Complex64,
    },
    /// Local transfer admittance combined with a surface admittance.
    TransferWithSurfaceAdmittance {
        /// Transfer admittance value
        transfer_admittance: Complex64,
        /// Surface admittance value
        surface_admittance: Complex64,
    },
}

impl BoundaryCondition {
    /// Get the boundary condition type index (matching ibval in C++)
    pub fn type_index(&self) -> i32 {
        match self {
            BoundaryCondition::Velocity(_) => 0,
            BoundaryCondition::Pressure(_) => 1,
            BoundaryCondition::VelocityWithAdmittance { .. } => 2,
            BoundaryCondition::TransferAdmittance { .. } => 3,
            BoundaryCondition::TransferWithSurfaceAdmittance { .. } => 4,
        }
    }

    /// Returns true if admittance is prescribed
    pub fn has_admittance(&self) -> bool {
        matches!(
            self,
            BoundaryCondition::VelocityWithAdmittance { .. }
                | BoundaryCondition::TransferAdmittance { .. }
                | BoundaryCondition::TransferWithSurfaceAdmittance { .. }
        )
    }

    /// Get admittance value if present
    pub fn admittance(&self) -> Option<Complex64> {
        match self {
            BoundaryCondition::VelocityWithAdmittance { admittance, .. } => Some(*admittance),
            BoundaryCondition::TransferAdmittance { admittance } => Some(*admittance),
            BoundaryCondition::TransferWithSurfaceAdmittance {
                transfer_admittance,
                surface_admittance,
            } => Some(*transfer_admittance + *surface_admittance),
            _ => None,
        }
    }
}

/// A boundary element
#[derive(Debug, Clone)]
pub struct Element {
    /// Global node indices (connectivity)
    pub connectivity: Vec<usize>,
    /// Element type (Tri3 or Quad4)
    pub element_type: ElementType,
    /// Element property (Surface, MidFace, Evaluation)
    pub property: ElementProperty,
    /// Unit normal vector at element center (3 components)
    pub normal: [f64; 3],
    /// Normal vectors at each node (for higher-order normals)
    pub node_normals: Vec<[f64; 3]>,
    /// Element center (centroid) coordinates
    pub center: [f64; 3],
    /// Element area
    pub area: f64,
    /// Boundary condition
    pub boundary_condition: BoundaryCondition,
    /// Element group index
    pub group: usize,
    /// DOF addresses (row numbers in coefficient matrix)
    pub dof_addresses: Vec<usize>,
}

impl Element {
    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.element_type.num_nodes()
    }

    /// Number of DOFs per element (typically 1 for constant elements)
    pub fn num_dofs(&self) -> usize {
        // For collocation BEM with constant elements
        1
    }

    /// Number of DOFs per element for a given basis order
    pub fn num_dofs_for_basis(&self, basis: BasisOrder) -> usize {
        self.element_type.num_dofs_for_basis(basis)
    }
}

/// Boundary element mesh
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Node coordinates (one [x, y, z] per node)
    pub nodes: Vec<[f64; 3]>,
    /// Elements
    pub elements: Vec<Element>,
    /// External node numbers (for output compatibility)
    pub external_node_numbers: Vec<i32>,
    /// External element numbers (for output compatibility)
    pub external_element_numbers: Vec<i32>,
    /// Number of boundary mesh nodes
    pub num_boundary_nodes: usize,
    /// Number of evaluation mesh nodes
    pub num_evaluation_nodes: usize,
    /// Number of boundary elements
    pub num_boundary_elements: usize,
    /// Number of evaluation elements
    pub num_evaluation_elements: usize,
    /// Symmetry planes (0 = none, +1 = symmetric, -1 = antisymmetric)
    pub symmetry_planes: [i32; 3],
    /// Symmetry plane coordinates
    pub symmetry_coordinates: [f64; 3],
    /// Number of element reflections due to symmetry
    pub num_reflections: usize,
    /// Node-to-element adjacency map: for each node, the list of element indices containing it.
    /// Built lazily via `build_adjacency()`.
    pub node_to_elements: Vec<Vec<usize>>,
}

impl Mesh {
    /// Total number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of elements
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Number of DOFs (rows in coefficient matrix) for constant elements
    pub fn num_dofs(&self) -> usize {
        // For collocation BEM: one DOF per surface element
        self.elements
            .iter()
            .filter(|e| e.property != ElementProperty::Evaluation)
            .count()
    }

    /// Number of DOFs for a given basis order
    pub fn num_dofs_for_basis(&self, basis: BasisOrder) -> usize {
        self.elements
            .iter()
            .filter(|e| e.property != ElementProperty::Evaluation)
            .map(|e| e.num_dofs_for_basis(basis))
            .sum()
    }

    /// Get node coordinates for an element
    pub fn element_nodes(&self, element: &Element) -> DenseMatrix<f64> {
        let n = element.num_nodes();
        let mut coords = DenseMatrix::<f64>::zeros(n, 3);
        for (i, &node_idx) in element.connectivity.iter().enumerate() {
            for j in 0..3 {
                coords[[i, j]] = self.nodes[node_idx][j];
            }
        }
        coords
    }

    /// Average element area
    pub fn average_element_area(&self) -> f64 {
        let surface_elements: Vec<_> = self
            .elements
            .iter()
            .filter(|e| e.property == ElementProperty::Surface)
            .collect();
        if surface_elements.is_empty() {
            return 0.0;
        }
        surface_elements.iter().map(|e| e.area).sum::<f64>() / surface_elements.len() as f64
    }

    /// Build the node-to-element adjacency map
    ///
    /// Populates `self.node_to_elements` so that `node_to_elements[node_idx]`
    /// contains the indices of all elements that reference that node.
    /// This is needed for efficient adjacency detection between elements.
    pub fn build_adjacency(&mut self) {
        self.node_to_elements = vec![Vec::new(); self.num_nodes()];
        for (elem_idx, elem) in self.elements.iter().enumerate() {
            for &node_idx in &elem.connectivity {
                self.node_to_elements[node_idx].push(elem_idx);
            }
        }
    }
}

// ============================================================================
// Element Adjacency
// ============================================================================

/// Adjacency type between two boundary elements
///
/// Describes the topological relationship between two elements,
/// which determines the integration method (Duffy, collocation-adjacent, or regular).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdjacencyType {
    /// Elements are the same (self-element integral)
    SelfElement,
    /// Elements share an edge (2 common nodes)
    CommonEdge {
        /// Local indices of the two shared nodes in the field element
        common_nodes: [usize; 2],
    },
    /// Elements share a vertex (1 common node)
    CommonVertex {
        /// Local index of the shared node in the field element
        common_node: usize,
    },
    /// Elements are separated (no common nodes)
    Separated,
}

/// Determine the adjacency type between two elements
///
/// Compares the global node indices in the connectivity arrays of both elements
/// to find common nodes. The result determines which integration method to use:
/// - `SelfElement`: same element indices -> Duffy transform
/// - `CommonEdge`: 2 shared nodes -> field-side collocation edge regularization
/// - `CommonVertex`: 1 shared node -> field-side collocation vertex regularization
/// - `Separated`: no shared nodes -> regular integration
pub fn detect_adjacency(elem_i: &Element, elem_j: &Element) -> AdjacencyType {
    // Check if same element (all nodes match)
    if elem_i.connectivity.len() == elem_j.connectivity.len()
        && elem_i
            .connectivity
            .iter()
            .all(|n| elem_j.connectivity.contains(n))
    {
        return AdjacencyType::SelfElement;
    }

    // Find common global node indices and their local positions in elem_j
    let mut common: Vec<usize> = Vec::new(); // local indices in elem_j
    for (j_local, &j_global) in elem_j.connectivity.iter().enumerate() {
        if elem_i.connectivity.contains(&j_global) {
            common.push(j_local);
        }
    }

    match common.len() {
        0 => AdjacencyType::Separated,
        1 => AdjacencyType::CommonVertex {
            common_node: common[0],
        },
        2 => AdjacencyType::CommonEdge {
            common_nodes: [common[0], common[1]],
        },
        // 3+ common nodes but not all matching: treat as self (degenerate case)
        _ => AdjacencyType::SelfElement,
    }
}

// ============================================================================
// Cluster Structures (for FMM)
// ============================================================================

/// A cluster of boundary elements (for Fast Multipole Method)
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Number of the element group to which the cluster belongs
    pub element_group: usize,
    /// Number of elements in the cluster
    pub num_elements: usize,
    /// Element property (Surface, MidFace, Evaluation)
    pub element_property: ElementProperty,
    /// True if all elements have the same number of nodes
    pub is_monotype: bool,
    /// True if admittance boundary conditions are prescribed
    pub has_admittance: bool,
    /// Number of unknown DOFs in the cluster
    pub num_dofs: usize,
    /// Number of DOFs per element
    pub dofs_per_element: usize,
    /// Element indices in this cluster
    pub element_indices: Vec<usize>,
    /// Center coordinates of the cluster
    pub center: [f64; 3],
    /// Radius of the cluster (distance from center to farthest element)
    pub radius: f64,
    /// Near-field cluster indices
    pub near_clusters: Vec<usize>,
    /// Far-field cluster indices
    pub far_clusters: Vec<usize>,
    /// Original cluster index (for reflections)
    pub original_cluster: usize,
    /// Reflection number (0-7)
    pub reflection_number: usize,
    /// Reflection factor (+1 or -1)
    pub reflection_factor: i32,
    /// Mirror image flag
    pub is_mirror: bool,
    /// Reflection directions [x, y, z]
    pub reflection_dirs: [bool; 3],
    /// Father cluster index (for multi-level)
    pub father: Option<usize>,
    /// Son cluster indices (for multi-level)
    pub sons: Vec<usize>,
    /// Level in the cluster tree
    pub level: usize,
    /// Fan-out clusters (far-field but fathers are near-field)
    pub fan_clusters: Vec<usize>,
}

impl Cluster {
    /// Create a new empty cluster
    pub fn new(center: [f64; 3]) -> Self {
        Self {
            element_group: 0,
            num_elements: 0,
            element_property: ElementProperty::Surface,
            is_monotype: true,
            has_admittance: false,
            num_dofs: 0,
            dofs_per_element: 1,
            element_indices: Vec::new(),
            center,
            radius: 0.0,
            near_clusters: Vec::new(),
            far_clusters: Vec::new(),
            original_cluster: 0,
            reflection_number: 0,
            reflection_factor: 1,
            is_mirror: false,
            reflection_dirs: [false, false, false],
            father: None,
            sons: Vec::new(),
            level: 0,
            fan_clusters: Vec::new(),
        }
    }
}

/// Parameters for a cluster tree level (for multi-level FMM)
#[derive(Debug, Clone)]
pub struct ClusterLevel {
    /// All clusters at this level
    pub clusters: Vec<Cluster>,
    /// Number of original (non-reflected) clusters
    pub num_original: usize,
    /// Maximum cluster radius at this level
    pub max_radius: f64,
    /// Average cluster radius
    pub avg_radius: f64,
    /// Minimum cluster radius
    pub min_radius: f64,
    /// Number of expansion terms for FMM
    pub expansion_terms: usize,
    /// Number of integration points on unit sphere
    pub sphere_points: usize,
    /// Number of theta direction points
    pub theta_points: usize,
    /// Number of phi direction points
    pub phi_points: usize,
    /// Gauss points coordinates in theta direction
    pub gauss_theta: Vec<f64>,
    /// Gauss weights in theta direction
    pub gauss_weights: Vec<f64>,
    /// Unit sphere quadrature points (x, y, z for each point)
    pub sphere_coords: DenseMatrix<f64>,
    /// Unit sphere quadrature weights
    pub sphere_weights: Vec<f64>,
}

impl ClusterLevel {
    /// Create a new cluster level with given expected capacity
    pub fn new(expected_clusters: usize) -> Self {
        Self {
            clusters: Vec::with_capacity(expected_clusters),
            num_original: 0,
            max_radius: 0.0,
            avg_radius: 0.0,
            min_radius: f64::MAX,
            expansion_terms: 4,
            sphere_points: 0,
            theta_points: 4,
            phi_points: 8,
            gauss_theta: Vec::new(),
            gauss_weights: Vec::new(),
            sphere_coords: DenseMatrix::<f64>::zeros(0, 3),
            sphere_weights: Vec::new(),
        }
    }
}

// ============================================================================
// Solver Configuration
// ============================================================================

/// BEM method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BemMethod {
    /// Traditional BEM (O(N²) dense matrix)
    Traditional = 0,
    /// Single-Level Fast Multipole BEM
    SingleLevelFmm = 1,
    /// Multi-Level Fast Multipole BEM
    MultiLevelFmm = 3,
}

/// Linear solver selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverMethod {
    /// Conjugate Gradient Squared
    Cgs = 0,
    /// Quasi-Minimal Residual CGSTAB
    QmrCgstab = 1,
    /// Stabilized BiCGSTAB
    BiCgstabStabilized = 2,
    /// BiCGSTAB
    BiCgstab = 3,
    /// Direct Gaussian elimination (LU factorization)
    Direct = 4,
}

/// Preconditioner selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Preconditioner {
    /// Incomplete LU, second order scanning
    IluSecondOrder = -2,
    /// Incomplete LU, first order scanning
    IluFirstOrder = -1,
    /// Incomplete LU, zero order scanning
    IluZeroOrder = 0,
    /// Diagonal scaling
    Scaling = 1,
    /// No preconditioner
    None = 2,
}

/// Solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// BEM method
    pub method: BemMethod,
    /// Linear solver
    pub solver: SolverMethod,
    /// Preconditioner
    pub preconditioner: Preconditioner,
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Minimum expansion terms for FMM
    pub min_expansion_terms: usize,
    /// Far-field cluster factor (typically 2/√3 ≈ 1.155)
    pub far_field_factor: f64,
    /// Average cluster edge length (0 = auto)
    pub cluster_edge_length: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            method: BemMethod::Traditional,
            solver: SolverMethod::Cgs,
            preconditioner: Preconditioner::IluZeroOrder,
            max_iterations: 250,
            tolerance: 1e-6,
            min_expansion_terms: 5,
            far_field_factor: 2.0 / 3.0_f64.sqrt(),
            cluster_edge_length: 0.0,
        }
    }
}

// ============================================================================
// Incident Waves
// ============================================================================

/// Incident plane wave
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaneWave {
    /// Direction of propagation (unit vector)
    pub direction: Vec<f64>,
    /// Amplitude (complex)
    pub amplitude: Complex64,
    /// Curve reference for frequency-dependent amplitude
    pub curve_ref: Option<usize>,
}

/// Point source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointSource {
    /// Location of the source
    pub position: Vec<f64>,
    /// Source strength (complex)
    pub strength: Complex64,
    /// Curve reference for frequency-dependent strength
    pub curve_ref: Option<usize>,
}

// ============================================================================
// Results
// ============================================================================

/// Solver result
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Solution vector (pressure or velocity potential at DOFs)
    pub solution: Vec<Complex64>,
    /// Number of iterations (for iterative solvers)
    pub iterations: usize,
    /// Final residual norm
    pub residual: f64,
    /// Converged flag
    pub converged: bool,
    /// Solve time in seconds
    pub solve_time: f64,
}

impl SolverResult {
    /// Create a converged result
    pub fn converged(solution: Vec<Complex64>, iterations: usize, residual: f64) -> Self {
        Self {
            solution,
            iterations,
            residual,
            converged: true,
            solve_time: 0.0,
        }
    }

    /// Create a non-converged result
    pub fn not_converged(solution: Vec<Complex64>, iterations: usize, residual: f64) -> Self {
        Self {
            solution,
            iterations,
            residual,
            converged: false,
            solve_time: 0.0,
        }
    }
}

/// Integration result for element kernels
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IntegrationResult {
    /// ∫ G dS (single layer potential)
    pub g_integral: Complex64,
    /// ∫ ∂G/∂n_y dS (double layer potential)
    pub dg_dn_integral: Complex64,
    /// ∫ ∂G/∂n_x dS (adjoint double layer)
    pub dg_dnx_integral: Complex64,
    /// ∫ ∂²G/(∂n_x ∂n_y) dS (hypersingular)
    pub d2g_dnxdny_integral: Complex64,
    /// Contribution to right-hand side
    pub rhs_contribution: Complex64,
}

/// Absolute and relative differences between two quadrature evaluations.
///
/// The estimate is computed component-wise from a coarse and a finer result.
/// It intentionally includes every kernel channel, including the hypersingular
/// channel, so callers cannot accidentally validate only the single-layer term.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IntegrationErrorEstimate {
    /// Absolute difference in the single-layer channel.
    pub g_absolute: f64,
    /// Relative difference in the single-layer channel.
    pub g_relative: f64,
    /// Absolute difference in the double-layer channel.
    pub dg_dn_absolute: f64,
    /// Relative difference in the double-layer channel.
    pub dg_dn_relative: f64,
    /// Absolute difference in the adjoint double-layer channel.
    pub dg_dnx_absolute: f64,
    /// Relative difference in the adjoint double-layer channel.
    pub dg_dnx_relative: f64,
    /// Absolute difference in the hypersingular channel.
    pub d2g_dnxdny_absolute: f64,
    /// Relative difference in the hypersingular channel.
    pub d2g_dnxdny_relative: f64,
    /// Absolute difference in the RHS contribution.
    pub rhs_absolute: f64,
    /// Relative difference in the RHS contribution.
    pub rhs_relative: f64,
    /// Largest absolute difference across all channels.
    pub max_absolute: f64,
    /// Largest relative difference across all channels.
    pub max_relative: f64,
}

impl IntegrationErrorEstimate {
    /// Return whether all channels satisfy a mixed absolute/relative tolerance.
    pub fn is_converged(
        &self,
        coarse: &IntegrationResult,
        fine: &IntegrationResult,
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> bool {
        if !absolute_tolerance.is_finite()
            || !relative_tolerance.is_finite()
            || absolute_tolerance < 0.0
            || relative_tolerance < 0.0
        {
            return false;
        }

        let differences = [
            self.g_absolute,
            self.dg_dn_absolute,
            self.dg_dnx_absolute,
            self.d2g_dnxdny_absolute,
            self.rhs_absolute,
        ];
        let coarse_values = [
            coarse.g_integral.norm(),
            coarse.dg_dn_integral.norm(),
            coarse.dg_dnx_integral.norm(),
            coarse.d2g_dnxdny_integral.norm(),
            coarse.rhs_contribution.norm(),
        ];
        let fine_values = [
            fine.g_integral.norm(),
            fine.dg_dn_integral.norm(),
            fine.dg_dnx_integral.norm(),
            fine.d2g_dnxdny_integral.norm(),
            fine.rhs_contribution.norm(),
        ];

        differences
            .into_iter()
            .zip(coarse_values.into_iter().zip(fine_values))
            .all(|(difference, (coarse_value, fine_value))| {
                difference.is_finite()
                    && coarse_value.is_finite()
                    && fine_value.is_finite()
                    && difference
                        <= absolute_tolerance + relative_tolerance * coarse_value.max(fine_value)
            })
    }
}

impl IntegrationResult {
    /// Estimate quadrature error by comparing this (coarse) result with a finer one.
    pub fn error_estimate(&self, finer: &Self) -> IntegrationErrorEstimate {
        let absolute = |coarse: Complex64, fine: Complex64| (fine - coarse).norm();
        let relative =
            |difference: f64, fine: Complex64| difference / fine.norm().max(f64::MIN_POSITIVE);

        let g_absolute = absolute(self.g_integral, finer.g_integral);
        let dg_dn_absolute = absolute(self.dg_dn_integral, finer.dg_dn_integral);
        let dg_dnx_absolute = absolute(self.dg_dnx_integral, finer.dg_dnx_integral);
        let d2g_dnxdny_absolute = absolute(self.d2g_dnxdny_integral, finer.d2g_dnxdny_integral);
        let rhs_absolute = absolute(self.rhs_contribution, finer.rhs_contribution);

        IntegrationErrorEstimate {
            g_absolute,
            g_relative: relative(g_absolute, finer.g_integral),
            dg_dn_absolute,
            dg_dn_relative: relative(dg_dn_absolute, finer.dg_dn_integral),
            dg_dnx_absolute,
            dg_dnx_relative: relative(dg_dnx_absolute, finer.dg_dnx_integral),
            d2g_dnxdny_absolute,
            d2g_dnxdny_relative: relative(d2g_dnxdny_absolute, finer.d2g_dnxdny_integral),
            rhs_absolute,
            rhs_relative: relative(rhs_absolute, finer.rhs_contribution),
            max_absolute: g_absolute
                .max(dg_dn_absolute)
                .max(dg_dnx_absolute)
                .max(d2g_dnxdny_absolute)
                .max(rhs_absolute),
            max_relative: relative(g_absolute, finer.g_integral)
                .max(relative(dg_dn_absolute, finer.dg_dn_integral))
                .max(relative(dg_dnx_absolute, finer.dg_dnx_integral))
                .max(relative(d2g_dnxdny_absolute, finer.d2g_dnxdny_integral))
                .max(relative(rhs_absolute, finer.rhs_contribution)),
        }
    }
}

/// Pair of adjacent quadrature evaluations and their component-wise error estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegrationEstimate {
    /// Lower-order (coarse) integration result.
    pub coarse: IntegrationResult,
    /// Higher-order (fine) integration result.
    pub fine: IntegrationResult,
    /// Difference between the coarse and fine results.
    pub error: IntegrationErrorEstimate,
}

impl IntegrationEstimate {
    /// Construct an estimate and calculate its component-wise error.
    pub fn from_results(coarse: IntegrationResult, fine: IntegrationResult) -> Self {
        let error = coarse.error_estimate(&fine);
        Self {
            coarse,
            fine,
            error,
        }
    }

    /// Return whether the coarse/fine pair passes a mixed tolerance gate.
    pub fn is_converged(&self, absolute_tolerance: f64, relative_tolerance: f64) -> bool {
        self.error.is_converged(
            &self.coarse,
            &self.fine,
            absolute_tolerance,
            relative_tolerance,
        )
    }

    /// Require the adjacent-order estimate to satisfy a numerical error
    /// budget before its finer result is consumed by a production path.
    pub fn require_converged(
        &self,
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> Result<(), QuadratureConvergenceError> {
        if !absolute_tolerance.is_finite()
            || !relative_tolerance.is_finite()
            || absolute_tolerance < 0.0
            || relative_tolerance < 0.0
        {
            return Err(QuadratureConvergenceError::InvalidTolerance {
                absolute_tolerance,
                relative_tolerance,
            });
        }
        if !self.is_converged(absolute_tolerance, relative_tolerance) {
            return Err(QuadratureConvergenceError::NotConverged {
                max_absolute: self.error.max_absolute,
                max_relative: self.error.max_relative,
                absolute_tolerance,
                relative_tolerance,
            });
        }
        Ok(())
    }
}

/// Failure while enforcing an adjacent-order quadrature error budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuadratureConvergenceError {
    /// Adjacent-order request was not strictly increasing.
    InvalidOrder(QuadratureOrderError),
    /// Absolute or relative tolerance is not finite and non-negative.
    InvalidTolerance {
        /// Requested absolute tolerance.
        absolute_tolerance: f64,
        /// Requested relative tolerance.
        relative_tolerance: f64,
    },
    /// The finer evaluation did not satisfy the requested mixed budget.
    NotConverged {
        /// Largest absolute adjacent-order difference.
        max_absolute: f64,
        /// Largest relative adjacent-order difference.
        max_relative: f64,
        /// Requested absolute tolerance.
        absolute_tolerance: f64,
        /// Requested relative tolerance.
        relative_tolerance: f64,
    },
}

impl std::fmt::Display for QuadratureConvergenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrder(error) => error.fmt(f),
            Self::InvalidTolerance {
                absolute_tolerance,
                relative_tolerance,
            } => write!(
                f,
                "quadrature tolerances must be finite and non-negative (absolute={absolute_tolerance}, relative={relative_tolerance})"
            ),
            Self::NotConverged {
                max_absolute,
                max_relative,
                absolute_tolerance,
                relative_tolerance,
            } => write!(
                f,
                "quadrature estimate not converged: max absolute={max_absolute:.3e}, max relative={max_relative:.3e}, budget=({absolute_tolerance:.3e}, {relative_tolerance:.3e})"
            ),
        }
    }
}

impl std::error::Error for QuadratureConvergenceError {}

/// Invalid adjacent-order request for a quadrature convergence estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadratureOrderError {
    /// Requested coarse quadrature order.
    pub coarse_order: usize,
    /// Requested fine quadrature order.
    pub fine_order: usize,
}

impl std::fmt::Display for QuadratureOrderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "quadrature fine order ({}) must be greater than coarse order ({})",
            self.fine_order, self.coarse_order
        )
    }
}

impl std::error::Error for QuadratureOrderError {}

/// Integration result for linear elements: one IntegrationResult per basis function
///
/// For a triangle element with 3 basis functions, this contains 3 IntegrationResults
/// where each result has the kernel integrals weighted by the corresponding shape function:
///   result[j].g_integral = ∫ N_j(y) G(x,y) dS_y
///   result[j].dg_dn_integral = ∫ N_j(y) ∂G/∂n_y(x,y) dS_y
///   etc.
#[derive(Debug, Clone)]
pub struct LinearIntegrationResult {
    /// Per-basis-function integration results
    pub basis_results: Vec<IntegrationResult>,
}

// VTU export: implement sonium_xem's MeshView so BEM binaries can pass
// `&mesh` directly to `sonium_xem::io::vtu::VtuWriter::new`. Only surface
// elements participate in the geometry; evaluation elements are skipped.
impl sonium_xem::io::vtu::MeshView for Mesh {
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.elements
            .iter()
            .filter(|e| !e.property.is_evaluation())
            .count()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.nodes.iter().copied())
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (sonium_xem::io::vtu::CellKind, Vec<u64>)> + '_> {
        use sonium_xem::io::vtu::CellKind;
        Box::new(
            self.elements
                .iter()
                .filter(|e| !e.property.is_evaluation())
                .map(|e| {
                    let kind = match e.element_type {
                        ElementType::Tri3 => CellKind::Triangle,
                        ElementType::Quad4 => CellKind::Quad,
                    };
                    let verts: Vec<u64> = e.connectivity.iter().map(|&i| i as u64).collect();
                    (kind, verts)
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_params() {
        let params = PhysicsParams::new(1000.0, 343.0, 1.21, false);
        assert!((params.wave_number - 2.0 * std::f64::consts::PI * 1000.0 / 343.0).abs() < 1e-10);
        assert!((params.wave_length - 0.343).abs() < 1e-10);
        assert_eq!(params.tau, 1.0);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn checked_physics_constructor_rejects_nonphysical_inputs() {
        assert!(matches!(
            PhysicsParams::try_new(0.0, 343.0, 1.21, false),
            Err(PhysicsParamsError::InvalidInput {
                field: "frequency",
                ..
            })
        ));
        assert!(matches!(
            PhysicsParams::try_new(1000.0, f64::NAN, 1.21, false),
            Err(PhysicsParamsError::InvalidInput {
                field: "speed_of_sound",
                ..
            })
        ));
    }

    #[test]
    fn typed_xem_quantities_reach_physics_without_unit_guessing() {
        let params = PhysicsParams::try_new_typed(
            sonium_xem::FrequencyHz::try_new(1_000.0).unwrap(),
            sonium_xem::SpeedOfSoundMPerS::try_new(343.0).unwrap(),
            sonium_xem::DensityKgM3::try_new(1.21).unwrap(),
            false,
        )
        .expect("validated XEM quantities are consumable by BEM");
        assert!((params.frequency - 1_000.0).abs() < f64::EPSILON);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn typed_acoustic_medium_propagates_sign_and_rejects_incompatible_conventions() {
        let mut conventions = sonium_xem::ConventionRegistry::default();
        conventions.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let medium = sonium_xem::AcousticMedium::try_new_with_conventions(
            sonium_xem::FrequencyHz::new(1_000.0),
            sonium_xem::SpeedOfSoundMPerS::new(343.0),
            sonium_xem::DensityKgM3::new(1.21),
            conventions,
        )
        .expect("valid medium");
        let params = PhysicsParams::try_new_medium(medium, false).expect("BEM consumes medium");
        assert_eq!(params.harmonic_factor, -1.0);

        conventions.outward_normals = false;
        let invalid = sonium_xem::AcousticMedium::try_new_with_conventions(
            sonium_xem::FrequencyHz::new(1_000.0),
            sonium_xem::SpeedOfSoundMPerS::new(343.0),
            sonium_xem::DensityKgM3::new(1.21),
            conventions,
        );
        assert!(invalid.is_err());
    }

    #[test]
    fn validation_rejects_tampered_derived_fields() {
        let mut params = PhysicsParams::new(1000.0, 343.0, 1.21, false);
        params.wave_number *= 1.01;
        assert!(matches!(
            params.validate(),
            Err(PhysicsParamsError::InconsistentDerived {
                field: "wave_number",
                ..
            })
        ));
    }

    #[test]
    fn explicit_exchange_convention_reaches_kernel_parameters() {
        let mut registry = sonium_xem::ConventionRegistry::default();
        registry.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let convention = BemConvention::from_exchange(&registry).unwrap();
        let params = PhysicsParams::new_with_convention(1000.0, 343.0, 1.21, false, convention);

        assert_eq!(params.harmonic_factor, -1.0);
        assert_eq!(params.pressure_factor, -1.21 * params.omega);
        assert_eq!(params.burton_miller_beta().im.signum(), -1.0);
    }

    #[test]
    fn exterior_formulation_sign_does_not_change_with_frequency() {
        for ka in [0.0, 0.49, 0.5, 1.0, 10.0] {
            assert_eq!(PhysicsParams::optimal_formulation_sign(ka), 1.0);
        }
    }

    #[test]
    fn adaptive_beta_uses_the_mie_validated_scale() {
        for ka in [0.1, 0.5, 1.0, 3.0, 10.0] {
            assert_eq!(PhysicsParams::optimal_beta_scale(ka), 0.2);
        }
    }

    #[test]
    fn test_element_type() {
        assert_eq!(ElementType::Tri3.num_nodes(), 3);
        assert_eq!(ElementType::Quad4.num_nodes(), 4);
    }

    #[test]
    fn test_boundary_condition() {
        let bc = BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]);
        assert_eq!(bc.type_index(), 0);
        assert!(!bc.has_admittance());

        let bc = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(1.0, 0.0)],
            admittance: Complex64::new(0.1, 0.0),
        };
        assert!(bc.has_admittance());
        assert_eq!(bc.admittance(), Some(Complex64::new(0.1, 0.0)));
    }

    #[test]
    fn test_cluster_creation() {
        let center = [0.0, 0.0, 0.0];
        let cluster = Cluster::new(center);
        assert_eq!(cluster.num_elements, 0);
        assert_eq!(cluster.reflection_factor, 1);
    }
}
