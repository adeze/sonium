//! Nodal DG solver for LEE on triangular/tetrahedral meshes.
//!
//! This solver advances linearized Euler perturbations about a uniform
//! mean flow on unstructured simplex meshes with a strong-form nodal
//! discontinuous Galerkin method of degree P1 through P3:
//!
//! ```text
//! ∫φ ∂q/∂t = −∫φ ∇·F(q) + ∫φ s + Σ_faces ∫φ ((F* − F)·n).
//! ```
//!
//! Inter-element coupling uses the exact upwind flux of the uniform-flow
//! LEE (closed-form `|A_n|` eigen-decomposition, no tuning parameter).
//! Boundary faces carry outflow, slip-wall, or prescribed-exterior
//! treatments per marker. Time integration is the three-stage SSPRK(3,3)
//! scheme with preallocated registers. Affine elements share one
//! reference mass inverse each, scaled by `1/detJ` per element.
//!
//! Stability follows the Cockburn--Shu estimate with the element
//! inscribed radius: `dt = CFL·h/((|U| + c)(2P + 1))` with `CFL ≤ 1`.
//!
//! # References
//!
//! J. S. Hesthaven and T. Warburton, *Nodal Discontinuous Galerkin
//! Methods*, Springer (2008).
//!
//! S. Gottlieb, C.-W. Shu, and E. Tadmor, "Strong Stability-Preserving
//! High-Order Time Discretization Methods", *SIAM Review* 43 (2001),
//! 89--112.

// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Mesh};
use crate::quadrature::{gauss_legendre_1d, gauss_tetrahedron, gauss_triangle};
use crate::spaces::FunctionSpace as _;
use crate::spaces::dg::{DGSpace, build_interior_facets};

use super::error::{AeroacousticsError, require_finite, require_positive};

/// Exterior state callback: position and time to `[p, ux, uy, uz]`.
///
/// Only the first `1 + dim` entries are read; the trailing entries are
/// ignored on two-dimensional meshes.
pub type ExteriorStateFn = fn(&[f64; 3], f64) -> [f64; 4];

/// Boundary treatment applied at one mesh marker.
#[derive(Debug, Clone, Copy)]
pub enum DgLeeBoundary {
    /// Far-field outflow: exterior is the undisturbed state (zero
    /// perturbation), so entering characteristics carry no incoming
    /// waves. Exact for normal-incidence plane waves; oblique waves
    /// reflect partially like any first-order absorbing condition.
    Outflow,
    /// Inviscid wall: exterior mirrors the normal velocity.
    SlipWall,
    /// Exact exterior state for inflow or manufactured solutions.
    Prescribed(ExteriorStateFn),
}

/// Configuration of an unstructured DG-LEE solver.
#[derive(Debug, Clone)]
pub struct DgLeeConfig {
    /// Polynomial degree (P1, P2, or P3).
    pub degree: PolynomialDegree,
    /// Uniform mean density `[kg/m^3]`.
    pub density: f64,
    /// Uniform sound speed `[m/s]`.
    pub speed_of_sound: f64,
    /// Uniform mean-flow velocity `[m/s]` (third entry must vanish in 2-D).
    pub mean_velocity: [f64; 3],
    /// Explicit SSPRK(3,3) step `[s]`.
    pub time_step: f64,
    /// Boundary treatment per marker (unlisted markers default to outflow).
    pub boundaries: HashMap<i32, DgLeeBoundary>,
}

impl DgLeeConfig {
    /// Construct a configuration and select `dt` from a CFL number.
    ///
    /// The step is `dt = CFL·h/((|U| + c)(2P + 1))` with the mesh
    /// inscribed radius `h`, so `CFL ≤ 1` is stable by construction.
    /// All markers default to outflow; refine with
    /// [`Self::with_boundary`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::basis::PolynomialDegree;
    /// use sonium_fem::mesh::unit_square_triangles;
    /// use sonium_fem::physics::acoustics::aeroacoustics::dg_lee::DgLeeConfig;
    ///
    /// let mesh = unit_square_triangles(2);
    /// let config =
    ///     DgLeeConfig::with_cfl(PolynomialDegree::P1, 1.2, 343.0, [30.0, 0.0, 0.0], &mesh, 0.5)
    ///         .unwrap();
    /// assert!(config.time_step > 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an unsupported mesh, a
    /// non-positive parameter, or a CFL number outside `(0, 1]`.
    pub fn with_cfl(
        degree: PolynomialDegree,
        density: f64,
        speed_of_sound: f64,
        mean_velocity: [f64; 3],
        mesh: &Mesh,
        cfl: f64,
    ) -> Result<Self, AeroacousticsError> {
        let candidate = Self {
            degree,
            density,
            speed_of_sound,
            mean_velocity,
            time_step: f64::NAN,
            boundaries: HashMap::new(),
        };
        candidate.validate_parameters(mesh)?;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        let rate = candidate.characteristic_rate(mesh)?;
        let mut configured = candidate;
        configured.time_step = cfl / rate;
        configured.validate(mesh)?;
        Ok(configured)
    }

    /// Assign a boundary treatment to one marker.
    pub fn with_boundary(mut self, marker: i32, boundary: DgLeeBoundary) -> Self {
        self.boundaries.insert(marker, boundary);
        self
    }

    /// Spectral-radius bound of the DG operator `[1/s]`.
    ///
    /// Combines the fastest characteristic with the `(2P + 1)` DG
    /// penalty over the mesh inscribed radius.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an unsupported or degenerate
    /// mesh, or non-physical flow parameters.
    pub fn characteristic_rate(&self, mesh: &Mesh) -> Result<f64, AeroacousticsError> {
        let admitted = admit_mesh(mesh, self.degree)?;
        require_positive("density", self.density)?;
        require_positive("speed_of_sound", self.speed_of_sound)?;
        for velocity in &self.mean_velocity {
            require_finite("mean_velocity", *velocity)?;
        }
        let speed = (self.mean_velocity[0].powi(2)
            + self.mean_velocity[1].powi(2)
            + self.mean_velocity[2].powi(2))
        .sqrt()
            + self.speed_of_sound;
        let h_min = min_inscribed_radius(mesh, &admitted)?;
        let penalty = 2.0 * self.degree.degree() as f64 + 1.0;
        let rate = speed * penalty / h_min;
        if !rate.is_finite() || rate <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "characteristic rate",
                value: rate,
            });
        }
        Ok(rate)
    }

    /// Validate parameters, mesh support, and the CFL contract.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for the same conditions as
    /// [`Self::with_cfl`], plus a time step violating `dt·rate ≤ 1`.
    pub fn validate(&self, mesh: &Mesh) -> Result<(), AeroacousticsError> {
        self.validate_parameters(mesh)?;
        require_positive("time_step", self.time_step)?;
        let rate = self.characteristic_rate(mesh)?;
        let cfl = self.time_step * rate;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        Ok(())
    }

    fn validate_parameters(&self, mesh: &Mesh) -> Result<(), AeroacousticsError> {
        let admitted = admit_mesh(mesh, self.degree)?;
        require_positive("density", self.density)?;
        require_positive("speed_of_sound", self.speed_of_sound)?;
        for velocity in &self.mean_velocity {
            require_finite("mean_velocity", *velocity)?;
        }
        if admitted.dim == 2 && self.mean_velocity[2] != 0.0 {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "a 2-D mesh cannot carry out-of-plane mean flow",
            });
        }
        Ok(())
    }
}

/// DG perturbation state in component-major layout.
///
/// Values pack as `values[(c·n_elements + e)·n_dof + i]` for component
/// `c`, element `e`, and local node `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct DgLeeState {
    /// Nodal values, component-major over the mesh.
    pub values: Vec<f64>,
    /// Component count (`1 + dim`).
    pub n_components: usize,
    /// Local nodes per element.
    pub dofs_per_element: usize,
    /// Element count.
    pub n_elements: usize,
}

impl DgLeeState {
    /// Construct a zero perturbation.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an unsupported mesh.
    pub fn zeros(config: &DgLeeConfig, mesh: &Mesh) -> Result<Self, AeroacousticsError> {
        let admitted = admit_mesh(mesh, config.degree)?;
        let n_components = admitted.dim + 1;
        Ok(Self {
            values: vec![0.0; n_components * mesh.num_elements() * admitted.n_dof],
            n_components,
            dofs_per_element: admitted.n_dof,
            n_elements: mesh.num_elements(),
        })
    }

    /// Construct the L2 projection of a stationary field.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an unsupported or degenerate
    /// mesh, or a non-finite projection.
    pub fn project<F>(
        config: &DgLeeConfig,
        mesh: &Mesh,
        field: F,
    ) -> Result<Self, AeroacousticsError>
    where
        F: Fn(&[f64; 3]) -> [f64; 4],
    {
        let operators = build_operators(config.degree, mesh)?;
        let mut state = Self::zeros(config, mesh)?;
        project_function(&operators, &mut state.values, field)?;
        Ok(state)
    }
}

/// Unstructured DG-LEE solver with SSPRK(3,3) stepping.
#[derive(Debug, Clone)]
pub struct DgLeeSolver {
    /// Discretization and mean-flow parameters.
    pub config: DgLeeConfig,
    /// Current perturbation state.
    pub state: DgLeeState,
    /// Simulated time `[s]`.
    pub time: f64,
    /// Volumetric source `(position, time) → [p, ux, uy, uz]` rate.
    pub source: Option<ExteriorStateFn>,
    operators: DgLeeOperators,
    stages: DgLeeStages,
}

impl DgLeeSolver {
    /// Construct a solver from a validated state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::basis::PolynomialDegree;
    /// use sonium_fem::mesh::unit_square_triangles;
    /// use sonium_fem::physics::acoustics::aeroacoustics::dg_lee::{
    ///     DgLeeConfig, DgLeeSolver, DgLeeState,
    /// };
    ///
    /// let mesh = unit_square_triangles(2);
    /// let config =
    ///     DgLeeConfig::with_cfl(PolynomialDegree::P1, 1.2, 343.0, [0.0, 0.0, 0.0], &mesh, 0.5)
    ///         .unwrap();
    /// let state = DgLeeState::zeros(&config, &mesh).unwrap();
    /// let mut solver = DgLeeSolver::new(config, &mesh, state).unwrap();
    /// solver.advance(2);
    /// assert!(solver.time > 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an invalid configuration, an
    /// unsupported or degenerate mesh, a mis-sized state, or a
    /// non-finite state.
    pub fn new(
        config: DgLeeConfig,
        mesh: &Mesh,
        state: DgLeeState,
    ) -> Result<Self, AeroacousticsError> {
        config.validate(mesh)?;
        let operators = build_operators(config.degree, mesh)?;
        let expected = operators.state_len();
        if state.values.len() != expected {
            return Err(AeroacousticsError::InvalidStateLength {
                expected,
                actual: state.values.len(),
            });
        }
        if state.values.iter().any(|x| !x.is_finite()) {
            return Err(AeroacousticsError::NonFiniteField { name: "DG state" });
        }
        let stages = DgLeeStages::zeros(expected);
        Ok(Self {
            config,
            state,
            time: 0.0,
            source: None,
            operators,
            stages,
        })
    }

    /// Advance one SSPRK(3,3) step.
    pub fn step(&mut self) {
        let dt = self.config.time_step;
        let time = self.time;
        let stages = &mut self.stages;
        rhs(
            &self.operators,
            &self.config,
            &self.state.values,
            time,
            self.source,
            &mut stages.load,
            &mut stages.derivative,
        );
        combine_ssprk(&self.state.values, &stages.derivative, dt, &mut stages.q1);
        rhs(
            &self.operators,
            &self.config,
            &stages.q1,
            time + dt,
            self.source,
            &mut stages.load,
            &mut stages.derivative,
        );
        combine_ssprk2(
            &self.state.values,
            &stages.q1,
            &stages.derivative,
            dt,
            &mut stages.q2,
        );
        rhs(
            &self.operators,
            &self.config,
            &stages.q2,
            time + 0.5 * dt,
            self.source,
            &mut stages.load,
            &mut stages.derivative,
        );
        combine_ssprk3_in_place(&mut self.state.values, &stages.q2, &stages.derivative, dt);
        self.time += dt;
    }

    /// Advance an integer number of SSPRK(3,3) steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

#[derive(Debug, Clone)]
struct DgLeeStages {
    q1: Vec<f64>,
    q2: Vec<f64>,
    load: Vec<f64>,
    derivative: Vec<f64>,
}

impl DgLeeStages {
    fn zeros(n: usize) -> Self {
        Self {
            q1: vec![0.0; n],
            q2: vec![0.0; n],
            load: vec![0.0; n],
            derivative: vec![0.0; n],
        }
    }
}

/// First SSPRK stage: `q1 = q + dt·L(q)`.
fn combine_ssprk(base: &[f64], derivative: &[f64], dt: f64, out: &mut [f64]) {
    for ((slot, x), k) in out.iter_mut().zip(base.iter()).zip(derivative.iter()) {
        *slot = x + dt * k;
    }
}

/// Second SSPRK stage: `q2 = 3q/4 + (q1 + dt·L(q1))/4`.
fn combine_ssprk2(base: &[f64], q1: &[f64], derivative: &[f64], dt: f64, out: &mut [f64]) {
    for (((slot, x), y), k) in out
        .iter_mut()
        .zip(base.iter())
        .zip(q1.iter())
        .zip(derivative.iter())
    {
        *slot = 0.75 * x + 0.25 * (y + dt * k);
    }
}

/// Third SSPRK stage, in place: `q = q/3 + 2(q2 + dt·L(q2))/3`.
fn combine_ssprk3_in_place(state: &mut [f64], q2: &[f64], derivative: &[f64], dt: f64) {
    for ((slot, y), k) in state.iter_mut().zip(q2.iter()).zip(derivative.iter()) {
        *slot = *slot / 3.0 + 2.0 * (y + dt * k) / 3.0;
    }
}

// ---------------------------------------------------------------------------
// Mesh admission and reference operators
// ---------------------------------------------------------------------------

/// Mesh validated for DG-LEE assembly.
#[derive(Debug, Clone, Copy)]
struct AdmittedMesh {
    dim: usize,
    element: ElementType,
    n_dof: usize,
}

/// Admit a simplex mesh of uniform supported type.
fn admit_mesh(mesh: &Mesh, degree: PolynomialDegree) -> Result<AdmittedMesh, AeroacousticsError> {
    if mesh.dimension != 2 && mesh.dimension != 3 {
        return Err(AeroacousticsError::UnsupportedDiscretization {
            detail: "DG-LEE needs a two- or three-dimensional mesh",
        });
    }
    if mesh.elements.is_empty() {
        return Err(AeroacousticsError::UnsupportedDiscretization {
            detail: "DG-LEE needs a non-empty mesh",
        });
    }
    let element = if mesh.dimension == 2 {
        ElementType::Triangle
    } else {
        ElementType::Tetrahedron
    };
    if mesh
        .elements
        .iter()
        .any(|candidate| candidate.element_type != element)
    {
        return Err(AeroacousticsError::UnsupportedDiscretization {
            detail: "DG-LEE needs uniform triangle or tetrahedron meshes",
        });
    }
    let space = DGSpace::new(degree);
    Ok(AdmittedMesh {
        dim: mesh.dimension,
        element,
        n_dof: space.num_local_dofs(element),
    })
}

/// Reference-element operators shared by all affine elements.
#[derive(Debug, Clone)]
struct ReferenceOperators {
    n_dof: usize,
    dim: usize,
    /// Inverse reference mass matrix, row-major.
    mass_inv: Vec<f64>,
    /// Basis values at volume quadrature, point-major.
    volume_values: Vec<f64>,
    /// Reference coordinates at volume quadrature.
    volume_coords: Vec<[f64; 3]>,
    /// Reference gradients at volume quadrature, point-major.
    volume_gradients: Vec<f64>,
    /// Volume weights summing to the reference volume.
    volume_weights: Vec<f64>,
    n_volume: usize,
    /// Face barycentric coordinates, point-major over sorted face nodes.
    face_barycentric: Vec<f64>,
    /// Face weights summing to the reference face measure.
    face_weights: Vec<f64>,
    n_face: usize,
    n_face_nodes: usize,
}

/// Quadrature order exact for mass products at one degree.
fn quadrature_order(degree: PolynomialDegree) -> usize {
    2 * degree.degree()
}

/// Reference vertex coordinates of the supported simplices.
fn reference_vertices(element: ElementType) -> Vec<[f64; 3]> {
    match element {
        ElementType::Triangle => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ElementType::Tetrahedron => vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        _ => unreachable!("admitted meshes hold simplices only"),
    }
}

/// Build reference operators for an admitted mesh and degree.
fn reference_operators(admitted: &AdmittedMesh, degree: PolynomialDegree) -> ReferenceOperators {
    let order = quadrature_order(degree);
    let volume_rule = match admitted.element {
        ElementType::Triangle => gauss_triangle(order),
        _ => gauss_tetrahedron(order),
    };
    let n_volume = volume_rule.len();
    let n_dof = admitted.n_dof;
    let mut volume_values = vec![0.0; n_volume * n_dof];
    let mut volume_coords = Vec::with_capacity(n_volume);
    let mut volume_gradients = vec![0.0; n_volume * n_dof * admitted.dim];
    let mut volume_weights = vec![0.0; n_volume];
    let mut mass = vec![0.0; n_dof * n_dof];
    for (q, point) in volume_rule.iter().enumerate() {
        let shape = crate::basis::evaluate_shape(
            admitted.element,
            degree,
            point.xi(),
            point.eta(),
            point.zeta(),
        );
        volume_coords.push(point.coords);
        volume_weights[q] = point.weight;
        for i in 0..n_dof {
            volume_values[q * n_dof + i] = shape.values[i];
            for d in 0..admitted.dim {
                volume_gradients[(q * n_dof + i) * admitted.dim + d] =
                    shape.gradients[i * shape.grad_dim + d];
            }
            for j in 0..n_dof {
                mass[i * n_dof + j] += point.weight * shape.values[i] * shape.values[j];
            }
        }
    }
    let (face_barycentric, face_weights, n_face_nodes) = match admitted.element {
        ElementType::Triangle => {
            // Gauss-Legendre on [-1, 1] mapped to barycentric pairs.
            let rule = gauss_legendre_1d(degree.degree() + 1);
            let mut barycentric = Vec::with_capacity(2 * rule.len());
            let mut weights = Vec::with_capacity(rule.len());
            for point in &rule {
                let s = (point.xi() + 1.0) * 0.5;
                barycentric.push(1.0 - s);
                barycentric.push(s);
                weights.push(point.weight);
            }
            (barycentric, weights, 2)
        }
        _ => {
            // Symmetric triangle rule reused on tetrahedral faces.
            let rule = gauss_triangle(order);
            let mut barycentric = Vec::with_capacity(3 * rule.len());
            let mut weights = Vec::with_capacity(rule.len());
            for point in &rule {
                barycentric.push(1.0 - point.xi() - point.eta());
                barycentric.push(point.xi());
                barycentric.push(point.eta());
                weights.push(point.weight);
            }
            (barycentric, weights, 3)
        }
    };
    let n_face = face_weights.len();
    ReferenceOperators {
        n_dof,
        dim: admitted.dim,
        mass_inv: invert_dense(&mass, n_dof),
        volume_values,
        volume_coords,
        volume_gradients,
        volume_weights,
        n_volume,
        face_barycentric,
        face_weights,
        n_face,
        n_face_nodes,
    }
}

/// Dense matrix inverse by Gauss-Jordan elimination with partial pivoting.
fn invert_dense(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut work = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            work[i * 2 * n + j] = matrix[i * n + j];
        }
        work[i * 2 * n + n + i] = 1.0;
    }
    for column in 0..n {
        let mut pivot = column;
        let mut magnitude = work[column * 2 * n + column].abs();
        for row in column + 1..n {
            let candidate = work[row * 2 * n + column].abs();
            if candidate > magnitude {
                magnitude = candidate;
                pivot = row;
            }
        }
        assert!(
            magnitude > 1e-14,
            "reference mass matrix must be positive definite"
        );
        if pivot != column {
            for k in 0..2 * n {
                work.swap(column * 2 * n + k, pivot * 2 * n + k);
            }
        }
        let diagonal = work[column * 2 * n + column];
        for k in 0..2 * n {
            work[column * 2 * n + k] /= diagonal;
        }
        for row in 0..n {
            if row != column {
                let factor = work[row * 2 * n + column];
                for k in 0..2 * n {
                    work[row * 2 * n + k] -= factor * work[column * 2 * n + k];
                }
            }
        }
    }
    let mut inverse = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inverse[i * n + j] = work[i * 2 * n + n + j];
        }
    }
    inverse
}

// ---------------------------------------------------------------------------
// Element geometry
// ---------------------------------------------------------------------------

/// Affine geometry of one element.
#[derive(Debug, Clone)]
struct ElementData {
    det_j: f64,
    /// Inverse-transpose Jacobian mapping reference to physical gradients.
    inv_jacobian_t: [[f64; 3]; 3],
}

/// Affine geometry of one admitted element.
fn element_data(
    mesh: &Mesh,
    elem_idx: usize,
    admitted: &AdmittedMesh,
) -> Result<ElementData, AeroacousticsError> {
    let vertices = mesh.elements[elem_idx].vertices();
    let origin = mesh.nodes[vertices[0]];
    let mut edges = [[0.0; 3]; 3];
    for (axis, edge) in edges.iter_mut().enumerate().take(admitted.dim) {
        let node = mesh.nodes[vertices[axis + 1]];
        edge[0] = node.x - origin.x;
        edge[1] = node.y - origin.y;
        edge[2] = node.z - origin.z;
    }
    // Jacobian columns are the vertex edges; the inverse-transpose maps
    // reference gradients to physical ones.
    let (det_j, inv_jacobian_t) = if admitted.dim == 2 {
        let det = edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0];
        // Inverse-transpose (not inverse): rows are [d, -c] and [-b, a].
        let inv_t = [
            [edges[1][1] / det, -edges[0][1] / det, 0.0],
            [-edges[1][0] / det, edges[0][0] / det, 0.0],
            [0.0, 0.0, 1.0],
        ];
        (det, inv_t)
    } else {
        let det = edges[0][0] * (edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1])
            - edges[0][1] * (edges[1][0] * edges[2][2] - edges[1][2] * edges[2][0])
            + edges[0][2] * (edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0]);
        // Adjugate-transpose over the determinant.
        let cofactor = |i: usize, j: usize| {
            let rows = [0, 1, 2].into_iter().filter(|row| *row != i);
            let cols = [0, 1, 2].into_iter().filter(|col| *col != j);
            let minor: Vec<f64> = rows
                .flat_map(|row| cols.clone().map(move |col| edges[col][row]))
                .collect();
            let sign = if (i + j).is_multiple_of(2) { 1.0 } else { -1.0 };
            sign * (minor[0] * minor[3] - minor[1] * minor[2])
        };
        let mut inv_t = [[0.0; 3]; 3];
        for (i, row) in inv_t.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = cofactor(i, j) / det;
            }
        }
        (det, inv_t)
    };
    if !det_j.is_finite() || det_j <= 0.0 {
        return Err(AeroacousticsError::DegenerateElement { index: elem_idx });
    }
    Ok(ElementData {
        det_j,
        inv_jacobian_t,
    })
}

/// Inscribed radius of one element (area/perimeter scaling).
fn inscribed_radius(mesh: &Mesh, elem_idx: usize, admitted: &AdmittedMesh) -> f64 {
    let vertices = mesh.elements[elem_idx].vertices();
    let points: Vec<[f64; 3]> = vertices
        .iter()
        .map(|node| {
            let point = mesh.nodes[*node];
            [point.x, point.y, point.z]
        })
        .collect();
    let edge_length = |a: usize, b: usize| {
        ((points[a][0] - points[b][0]).powi(2)
            + (points[a][1] - points[b][1]).powi(2)
            + (points[a][2] - points[b][2]).powi(2))
        .sqrt()
    };
    if admitted.dim == 2 {
        let area = 0.5
            * ((points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
                - (points[2][0] - points[0][0]) * (points[1][1] - points[0][1]))
                .abs();
        let perimeter = edge_length(0, 1) + edge_length(1, 2) + edge_length(2, 0);
        2.0 * area / perimeter
    } else {
        let volume = ((points[1][0] - points[0][0])
            * ((points[2][1] - points[0][1]) * (points[3][2] - points[0][2])
                - (points[2][2] - points[0][2]) * (points[3][1] - points[0][1]))
            - (points[1][1] - points[0][1])
                * ((points[2][0] - points[0][0]) * (points[3][2] - points[0][2])
                    - (points[2][2] - points[0][2]) * (points[3][0] - points[0][0]))
            + (points[1][2] - points[0][2])
                * ((points[2][0] - points[0][0]) * (points[3][1] - points[0][1])
                    - (points[2][1] - points[0][1]) * (points[3][0] - points[0][0])))
            .abs()
            / 6.0;
        let mut surface = 0.0;
        for face in admitted.element.local_faces() {
            let a = points[face[0]];
            let b = points[face[1]];
            let c = points[face[2]];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            surface += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        3.0 * volume / surface
    }
}

/// Minimum inscribed radius over the mesh.
fn min_inscribed_radius(mesh: &Mesh, admitted: &AdmittedMesh) -> Result<f64, AeroacousticsError> {
    let mut h_min = f64::INFINITY;
    for elem_idx in 0..mesh.num_elements() {
        // Geometry errors surface here so CFL admission never divides by a
        // degenerate element size.
        element_data(mesh, elem_idx, admitted)?;
        h_min = h_min.min(inscribed_radius(mesh, elem_idx, admitted));
    }
    if !h_min.is_finite() || h_min <= 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "mesh size",
            value: h_min,
        });
    }
    Ok(h_min)
}

// ---------------------------------------------------------------------------
// Facet construction
// ---------------------------------------------------------------------------

/// Basis traces of one facet side at the face quadrature points.
#[derive(Debug, Clone)]
struct FacetSideData {
    element: usize,
    /// Basis values, point-major over face quadrature.
    basis: Vec<f64>,
}

/// Interior facet with oriented flux data.
#[derive(Debug, Clone)]
struct InteriorFacetData {
    plus: FacetSideData,
    minus: FacetSideData,
    /// Unit normal pointing from the plus to the minus element.
    normal: [f64; 3],
    weights: Vec<f64>,
}

/// Boundary facet with oriented flux data.
#[derive(Debug, Clone)]
struct BoundaryFacetData {
    side: FacetSideData,
    /// Outward unit normal.
    normal: [f64; 3],
    marker: i32,
    points: Vec<[f64; 3]>,
    weights: Vec<f64>,
}

/// Element centroid from its vertices.
fn element_centroid(mesh: &Mesh, elem_idx: usize) -> [f64; 3] {
    let vertices = mesh.elements[elem_idx].vertices();
    let mut centroid = [0.0; 3];
    for node in vertices {
        let point = mesh.nodes[*node];
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
    }
    let count = vertices.len() as f64;
    [
        centroid[0] / count,
        centroid[1] / count,
        centroid[2] / count,
    ]
}

/// Unit normal of a facet oriented from `from` toward `to`.
fn oriented_normal(nodes: &[[f64; 3]], from: [f64; 3], to: [f64; 3]) -> [f64; 3] {
    let mut normal = if nodes.len() == 2 {
        let tangent = [
            nodes[1][0] - nodes[0][0],
            nodes[1][1] - nodes[0][1],
            nodes[1][2] - nodes[0][2],
        ];
        [tangent[1], -tangent[0], 0.0]
    } else {
        let u = [
            nodes[1][0] - nodes[0][0],
            nodes[1][1] - nodes[0][1],
            nodes[1][2] - nodes[0][2],
        ];
        let v = [
            nodes[2][0] - nodes[0][0],
            nodes[2][1] - nodes[0][1],
            nodes[2][2] - nodes[0][2],
        ];
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let norm = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
    normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
    let direction = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    if normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2] < 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    normal
}

/// Facet area from its sorted corner nodes.
fn facet_area_from_nodes(nodes: &[[f64; 3]]) -> f64 {
    if nodes.len() == 2 {
        ((nodes[1][0] - nodes[0][0]).powi(2)
            + (nodes[1][1] - nodes[0][1]).powi(2)
            + (nodes[1][2] - nodes[0][2]).powi(2))
        .sqrt()
    } else {
        let u = [
            nodes[1][0] - nodes[0][0],
            nodes[1][1] - nodes[0][1],
            nodes[1][2] - nodes[0][2],
        ];
        let v = [
            nodes[2][0] - nodes[0][0],
            nodes[2][1] - nodes[0][1],
            nodes[2][2] - nodes[0][2],
        ];
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt()
    }
}

/// Local vertex indices of sorted face nodes within one element.
fn local_face_indices(mesh: &Mesh, elem_idx: usize, sorted: &[usize]) -> Vec<usize> {
    let vertices = mesh.elements[elem_idx].vertices();
    sorted
        .iter()
        .map(|node| {
            vertices
                .iter()
                .position(|vertex| vertex == node)
                .expect("facet node belongs to its element")
        })
        .collect()
}

/// Basis traces of one element at face quadrature points.
///
/// Face quadrature is parametrized by the facet's globally sorted
/// corner nodes, so both adjacent elements sample identical physical
/// points regardless of their local face orientation.
fn side_basis(
    mesh: &Mesh,
    admitted: &AdmittedMesh,
    degree: PolynomialDegree,
    reference: &ReferenceOperators,
    elem_idx: usize,
    sorted: &[usize],
) -> Vec<f64> {
    let local = local_face_indices(mesh, elem_idx, sorted);
    let ref_vertices = reference_vertices(admitted.element);
    let mut basis = vec![0.0; reference.n_face * admitted.n_dof];
    for q in 0..reference.n_face {
        let mut coords = [0.0; 3];
        for (bary, local_idx) in reference.face_barycentric
            [q * reference.n_face_nodes..(q + 1) * reference.n_face_nodes]
            .iter()
            .zip(local.iter())
        {
            let vertex = ref_vertices[*local_idx];
            coords[0] += bary * vertex[0];
            coords[1] += bary * vertex[1];
            coords[2] += bary * vertex[2];
        }
        let shape =
            crate::basis::evaluate_shape(admitted.element, degree, coords[0], coords[1], coords[2]);
        for i in 0..admitted.n_dof {
            basis[q * admitted.n_dof + i] = shape.values[i];
        }
    }
    basis
}

/// Physical face quadrature points over sorted corner nodes.
fn face_points(nodes: &[[f64; 3]], reference: &ReferenceOperators) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(reference.n_face);
    for q in 0..reference.n_face {
        let mut point = [0.0; 3];
        for (bary, corner) in reference.face_barycentric
            [q * reference.n_face_nodes..(q + 1) * reference.n_face_nodes]
            .iter()
            .zip(nodes.iter())
        {
            point[0] += bary * corner[0];
            point[1] += bary * corner[1];
            point[2] += bary * corner[2];
        }
        points.push(point);
    }
    points
}

/// Corner coordinates of sorted facet nodes.
fn sorted_corner_coords(mesh: &Mesh, sorted: &[usize]) -> Vec<[f64; 3]> {
    sorted
        .iter()
        .map(|node| {
            let point = mesh.nodes[*node];
            [point.x, point.y, point.z]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Operator assembly
// ---------------------------------------------------------------------------

/// Precomputed DG operators and geometry.
#[derive(Debug, Clone)]
struct DgLeeOperators {
    admitted: AdmittedMesh,
    reference: ReferenceOperators,
    elements: Vec<ElementData>,
    interior: Vec<InteriorFacetData>,
    boundary: Vec<BoundaryFacetData>,
    /// Physical volume quadrature points, element-major.
    volume_points: Vec<[f64; 3]>,
    /// Physical volume weights, element-major.
    volume_weights: Vec<f64>,
}

impl DgLeeOperators {
    fn state_len(&self) -> usize {
        (self.admitted.dim + 1) * self.elements.len() * self.admitted.n_dof
    }

    fn component_base(&self, component: usize, element: usize) -> usize {
        (component * self.elements.len() + element) * self.admitted.n_dof
    }
}

/// Physical volume quadrature of one element under the P1 map.
fn volume_quadrature(
    mesh: &Mesh,
    elem_idx: usize,
    admitted: &AdmittedMesh,
    reference: &ReferenceOperators,
    det_j: f64,
) -> (Vec<[f64; 3]>, Vec<f64>) {
    let vertices = mesh.elements[elem_idx].vertices();
    let corners: Vec<[f64; 3]> = vertices
        .iter()
        .map(|node| {
            let point = mesh.nodes[*node];
            [point.x, point.y, point.z]
        })
        .collect();
    let mut points = Vec::with_capacity(reference.n_volume);
    let mut weights = Vec::with_capacity(reference.n_volume);
    for (coords, weight) in reference
        .volume_coords
        .iter()
        .zip(reference.volume_weights.iter())
    {
        let mut mapped = corners[0];
        for (axis, corner) in corners.iter().enumerate().skip(1).take(admitted.dim) {
            let coordinate = coords[axis - 1];
            mapped[0] += coordinate * (corner[0] - corners[0][0]);
            mapped[1] += coordinate * (corner[1] - corners[0][1]);
            mapped[2] += coordinate * (corner[2] - corners[0][2]);
        }
        points.push(mapped);
        weights.push(weight * det_j);
    }
    (points, weights)
}

/// Build all DG operators for a mesh and degree.
fn build_operators(
    degree: PolynomialDegree,
    mesh: &Mesh,
) -> Result<DgLeeOperators, AeroacousticsError> {
    let admitted = admit_mesh(mesh, degree)?;
    let reference = reference_operators(&admitted, degree);
    let mut elements = Vec::with_capacity(mesh.num_elements());
    for elem_idx in 0..mesh.num_elements() {
        elements.push(element_data(mesh, elem_idx, &admitted)?);
    }
    let ref_face_measure = reference.face_weights.iter().sum::<f64>();
    let mut interior = Vec::new();
    for facet in build_interior_facets(mesh) {
        let mut sorted = facet.shared_nodes.clone();
        sorted.sort_unstable();
        let corners = sorted_corner_coords(mesh, &sorted);
        let area = facet_area_from_nodes(&corners);
        let normal = oriented_normal(
            &corners,
            element_centroid(mesh, facet.elem_plus),
            element_centroid(mesh, facet.elem_minus),
        );
        let scale = area / ref_face_measure;
        interior.push(InteriorFacetData {
            plus: FacetSideData {
                element: facet.elem_plus,
                basis: side_basis(
                    mesh,
                    &admitted,
                    degree,
                    &reference,
                    facet.elem_plus,
                    &sorted,
                ),
            },
            minus: FacetSideData {
                element: facet.elem_minus,
                basis: side_basis(
                    mesh,
                    &admitted,
                    degree,
                    &reference,
                    facet.elem_minus,
                    &sorted,
                ),
            },
            normal,
            weights: reference.face_weights.iter().map(|w| w * scale).collect(),
        });
    }
    let mut boundary = Vec::new();
    for face in &mesh.boundaries {
        let mut sorted = face.nodes.clone();
        sorted.sort_unstable();
        let corners = sorted_corner_coords(mesh, &sorted);
        let area = facet_area_from_nodes(&corners);
        let centroid = element_centroid(mesh, face.element_idx);
        let face_centroid = [
            corners.iter().map(|corner| corner[0]).sum::<f64>() / corners.len() as f64,
            corners.iter().map(|corner| corner[1]).sum::<f64>() / corners.len() as f64,
            corners.iter().map(|corner| corner[2]).sum::<f64>() / corners.len() as f64,
        ];
        let normal = oriented_normal(&corners, centroid, face_centroid);
        let scale = area / ref_face_measure;
        boundary.push(BoundaryFacetData {
            side: FacetSideData {
                element: face.element_idx,
                basis: side_basis(
                    mesh,
                    &admitted,
                    degree,
                    &reference,
                    face.element_idx,
                    &sorted,
                ),
            },
            normal,
            marker: face.marker,
            points: face_points(&corners, &reference),
            weights: reference.face_weights.iter().map(|w| w * scale).collect(),
        });
    }
    let mut volume_points = Vec::new();
    let mut volume_weights = Vec::new();
    for (elem_idx, element) in elements.iter().enumerate() {
        let (mut points, mut weights) =
            volume_quadrature(mesh, elem_idx, &admitted, &reference, element.det_j);
        volume_points.append(&mut points);
        volume_weights.append(&mut weights);
    }
    Ok(DgLeeOperators {
        admitted,
        reference,
        elements,
        interior,
        boundary,
        volume_points,
        volume_weights,
    })
}

// ---------------------------------------------------------------------------
// Upwind flux
// ---------------------------------------------------------------------------

/// Normal flux `F(q)·n` of the uniform-flow LEE.
fn physical_flux_normal(
    state: &[f64],
    normal: &[f64; 3],
    dim: usize,
    flow: &[f64; 3],
    rho: f64,
    c: f64,
    out: &mut [f64],
) {
    let normal_flow = flow[0] * normal[0] + flow[1] * normal[1] + flow[2] * normal[2];
    let mut normal_velocity = 0.0;
    for d in 0..dim {
        normal_velocity += state[1 + d] * normal[d];
    }
    out[0] = normal_flow * state[0] + rho * c * c * normal_velocity;
    for d in 0..dim {
        out[1 + d] = normal_flow * state[1 + d] + state[0] * normal[d] / rho;
    }
}

/// Exact upwind flux `F*(q_in, q_out, n)` for the uniform-flow LEE.
///
/// Central average minus half the `|A_n|`-weighted jump, with the
/// dissipation assembled from the closed-form eigen-decomposition:
/// acoustic waves `|U_n ± c|` on `[±ρc, n]` plus transverse convection
/// `|U_n|` on the normal velocity complement. The normal must be a unit
/// vector within the velocity space (all facet normals qualify).
fn upwind_flux(
    interior: &[f64],
    exterior: &[f64],
    normal: &[f64; 3],
    dim: usize,
    flow: &[f64; 3],
    rho: f64,
    c: f64,
    out: &mut [f64],
) {
    let mut average = [0.0; 4];
    let mut flux_in = [0.0; 4];
    let mut flux_out = [0.0; 4];
    physical_flux_normal(interior, normal, dim, flow, rho, c, &mut flux_in);
    physical_flux_normal(exterior, normal, dim, flow, rho, c, &mut flux_out);
    for component in 0..dim + 1 {
        average[component] = 0.5 * (flux_in[component] + flux_out[component]);
    }
    let normal_flow = flow[0] * normal[0] + flow[1] * normal[1] + flow[2] * normal[2];
    let pressure_jump = exterior[0] - interior[0];
    let mut normal_jump = 0.0;
    for d in 0..dim {
        normal_jump += (exterior[1 + d] - interior[1 + d]) * normal[d];
    }
    let alpha_plus = 0.5 * (normal_jump + pressure_jump / (rho * c));
    let alpha_minus = 0.5 * (normal_jump - pressure_jump / (rho * c));
    let downwind = (normal_flow + c).abs();
    let upwind = (normal_flow - c).abs();
    let transverse = normal_flow.abs();
    out[0] =
        average[0] - 0.5 * (upwind * alpha_minus * (-rho * c) + downwind * alpha_plus * rho * c);
    for d in 0..dim {
        let jump = exterior[1 + d] - interior[1 + d];
        let transverse_jump = jump - normal_jump * normal[d];
        out[1 + d] = average[1 + d]
            - 0.5
                * (upwind * alpha_minus * normal[d]
                    + transverse * transverse_jump
                    + downwind * alpha_plus * normal[d]);
    }
}

// ---------------------------------------------------------------------------
// Right-hand side
// ---------------------------------------------------------------------------

/// Evaluate one component and its physical gradient at volume quadrature.
fn evaluate_volume(
    operators: &DgLeeOperators,
    values: &[f64],
    component: usize,
    element: usize,
    inv_jacobian_t: &[[f64; 3]; 3],
    out_values: &mut [f64],
    out_gradients: &mut [[f64; 3]],
) {
    let reference = &operators.reference;
    let base = operators.component_base(component, element);
    for q in 0..reference.n_volume {
        let mut value = 0.0;
        let mut gradient = [0.0; 3];
        for i in 0..reference.n_dof {
            let dof = values[base + i];
            value += dof * reference.volume_values[q * reference.n_dof + i];
            for (d, slot) in gradient.iter_mut().take(reference.dim).enumerate() {
                *slot +=
                    dof * reference.volume_gradients[(q * reference.n_dof + i) * reference.dim + d];
            }
        }
        // Physical gradient via the inverse-transpose Jacobian.
        let mut physical = [0.0; 3];
        for row in 0..3 {
            physical[row] = inv_jacobian_t[row][0] * gradient[0]
                + inv_jacobian_t[row][1] * gradient[1]
                + inv_jacobian_t[row][2] * gradient[2];
        }
        out_values[q] = value;
        out_gradients[q] = physical;
    }
}

/// Evaluate one component at face quadrature from side traces.
fn evaluate_face(
    operators: &DgLeeOperators,
    values: &[f64],
    component: usize,
    side: &FacetSideData,
    out: &mut [f64],
) {
    let reference = &operators.reference;
    let base = operators.component_base(component, side.element);
    for (q, slot) in out.iter_mut().enumerate() {
        let mut value = 0.0;
        for i in 0..reference.n_dof {
            value += values[base + i] * side.basis[q * reference.n_dof + i];
        }
        *slot = value;
    }
}

/// Strong-form DG right-hand side with the volumetric source.
fn rhs(
    operators: &DgLeeOperators,
    config: &DgLeeConfig,
    values: &[f64],
    time: f64,
    source: Option<ExteriorStateFn>,
    load: &mut [f64],
    output: &mut [f64],
) {
    load.fill(0.0);
    let reference = &operators.reference;
    let dim = reference.dim;
    let n_components = dim + 1;
    let rho = config.density;
    let c = config.speed_of_sound;
    let flow = config.mean_velocity;
    let mut point_values = vec![0.0; n_components * reference.n_volume];
    let mut point_gradients = vec![[0.0; 3]; n_components * reference.n_volume];
    let mut divergence = vec![0.0; n_components * reference.n_volume];

    // Volume terms: strong-form divergence plus the volumetric source.
    for (element, geometry) in operators.elements.iter().enumerate() {
        for component in 0..n_components {
            let value_range = component * reference.n_volume..(component + 1) * reference.n_volume;
            let gradient_range =
                component * reference.n_volume..(component + 1) * reference.n_volume;
            evaluate_volume(
                operators,
                values,
                component,
                element,
                &geometry.inv_jacobian_t,
                &mut point_values[value_range],
                &mut point_gradients[gradient_range],
            );
        }
        for q in 0..reference.n_volume {
            let mut velocity_divergence = 0.0;
            for d in 0..dim {
                velocity_divergence += point_gradients[(1 + d) * reference.n_volume + q][d];
            }
            let mut convection = [0.0; 4];
            for component in 0..n_components {
                let gradient = point_gradients[component * reference.n_volume + q];
                convection[component] =
                    flow[0] * gradient[0] + flow[1] * gradient[1] + flow[2] * gradient[2];
            }
            divergence[q] = convection[0] + rho * c * c * velocity_divergence;
            for d in 0..dim {
                divergence[(1 + d) * reference.n_volume + q] =
                    convection[1 + d] + point_gradients[q][d] / rho;
            }
        }
        if let Some(source_fn) = source {
            for q in 0..reference.n_volume {
                let position = operators.volume_points[element * reference.n_volume + q];
                let rate = source_fn(&position, time);
                for component in 0..n_components {
                    divergence[component * reference.n_volume + q] -= rate[component];
                }
            }
        }
        for component in 0..n_components {
            let base = operators.component_base(component, element);
            for q in 0..reference.n_volume {
                let weight = operators.volume_weights[element * reference.n_volume + q];
                let residual = divergence[component * reference.n_volume + q];
                for i in 0..reference.n_dof {
                    load[base + i] -=
                        weight * reference.volume_values[q * reference.n_dof + i] * residual;
                }
            }
        }
    }

    // Interior faces: jump correction on both sides.
    let mut trace_in = vec![0.0; n_components * reference.n_face];
    let mut trace_out = vec![0.0; n_components * reference.n_face];
    for facet in &operators.interior {
        for component in 0..n_components {
            evaluate_face(
                operators,
                values,
                component,
                &facet.plus,
                &mut trace_in[component * reference.n_face..(component + 1) * reference.n_face],
            );
            evaluate_face(
                operators,
                values,
                component,
                &facet.minus,
                &mut trace_out[component * reference.n_face..(component + 1) * reference.n_face],
            );
        }
        correct_side(
            operators,
            config,
            facet.weights.as_slice(),
            &facet.plus,
            &facet.normal,
            &trace_in,
            &trace_out,
            load,
        );
        let opposite = [-facet.normal[0], -facet.normal[1], -facet.normal[2]];
        correct_side(
            operators,
            config,
            facet.weights.as_slice(),
            &facet.minus,
            &opposite,
            &trace_out,
            &trace_in,
            load,
        );
    }

    // Boundary faces per marker treatment.
    for facet in &operators.boundary {
        for component in 0..n_components {
            evaluate_face(
                operators,
                values,
                component,
                &facet.side,
                &mut trace_in[component * reference.n_face..(component + 1) * reference.n_face],
            );
        }
        match config
            .boundaries
            .get(&facet.marker)
            .copied()
            .unwrap_or(DgLeeBoundary::Outflow)
        {
            // Free-stream exterior: entering characteristics see the
            // undisturbed state (no incoming waves). Copying the interior
            // trace instead would feed interior values back through the
            // entering characteristics, an ill-posed feedback that grows
            // at higher orders.
            DgLeeBoundary::Outflow => {
                for slot in trace_out.iter_mut() {
                    *slot = 0.0;
                }
                correct_side(
                    operators,
                    config,
                    facet.weights.as_slice(),
                    &facet.side,
                    &facet.normal,
                    &trace_in,
                    &trace_out,
                    load,
                );
            }
            DgLeeBoundary::SlipWall => {
                for q in 0..reference.n_face {
                    let mut normal_velocity = 0.0;
                    for d in 0..dim {
                        normal_velocity +=
                            trace_in[(1 + d) * reference.n_face + q] * facet.normal[d];
                    }
                    trace_out[q] = trace_in[q];
                    for d in 0..dim {
                        trace_out[(1 + d) * reference.n_face + q] = trace_in
                            [(1 + d) * reference.n_face + q]
                            - 2.0 * normal_velocity * facet.normal[d];
                    }
                }
                correct_side(
                    operators,
                    config,
                    facet.weights.as_slice(),
                    &facet.side,
                    &facet.normal,
                    &trace_in,
                    &trace_out,
                    load,
                );
            }
            DgLeeBoundary::Prescribed(exterior) => {
                for (q, point) in facet.points.iter().enumerate() {
                    let state = exterior(point, time);
                    for component in 0..n_components {
                        trace_out[component * reference.n_face + q] = state[component];
                    }
                }
                correct_side(
                    operators,
                    config,
                    facet.weights.as_slice(),
                    &facet.side,
                    &facet.normal,
                    &trace_in,
                    &trace_out,
                    load,
                );
            }
        }
    }

    // Element-local mass solve: shared reference inverse over detJ.
    for element in 0..operators.elements.len() {
        let det_j = operators.elements[element].det_j;
        for component in 0..n_components {
            let base = operators.component_base(component, element);
            for i in 0..reference.n_dof {
                let mut value = 0.0;
                for j in 0..reference.n_dof {
                    value += reference.mass_inv[i * reference.n_dof + j] * load[base + j];
                }
                output[base + i] = value / det_j;
            }
        }
    }
}

/// Scatter one side's flux correction into the load vector.
fn correct_side(
    operators: &DgLeeOperators,
    config: &DgLeeConfig,
    weights: &[f64],
    side: &FacetSideData,
    normal: &[f64; 3],
    trace_in: &[f64],
    trace_out: &[f64],
    load: &mut [f64],
) {
    let reference = &operators.reference;
    let dim = reference.dim;
    let n_components = dim + 1;
    let mut state_in = [0.0; 4];
    let mut state_out = [0.0; 4];
    let mut flux_star = [0.0; 4];
    let mut flux_interior = [0.0; 4];
    for q in 0..reference.n_face {
        for component in 0..n_components {
            state_in[component] = trace_in[component * reference.n_face + q];
            state_out[component] = trace_out[component * reference.n_face + q];
        }
        upwind_flux(
            &state_in,
            &state_out,
            normal,
            dim,
            &config.mean_velocity,
            config.density,
            config.speed_of_sound,
            &mut flux_star,
        );
        physical_flux_normal(
            &state_in,
            normal,
            dim,
            &config.mean_velocity,
            config.density,
            config.speed_of_sound,
            &mut flux_interior,
        );
        // Strong-form surface correction +(F^- - F*).n: integration by
        // parts on the volume term (-int phi div F = +int grad phi.F -
        // int phi F^-.n) forces the minus sign so the strong operator
        // matches the weak form int grad phi.F - int phi F*.n.
        for component in 0..n_components {
            let base = operators.component_base(component, side.element);
            let jump = flux_interior[component] - flux_star[component];
            for i in 0..reference.n_dof {
                load[base + i] += weights[q] * side.basis[q * reference.n_dof + i] * jump;
            }
        }
    }
}

/// L2 projection of a stationary field onto the DG space.
fn project_function<F>(
    operators: &DgLeeOperators,
    values: &mut [f64],
    field: F,
) -> Result<(), AeroacousticsError>
where
    F: Fn(&[f64; 3]) -> [f64; 4],
{
    let reference = &operators.reference;
    let dim = reference.dim;
    for element in 0..operators.elements.len() {
        let det_j = operators.elements[element].det_j;
        for component in 0..dim + 1 {
            let base = operators.component_base(component, element);
            let mut load = vec![0.0; reference.n_dof];
            for q in 0..reference.n_volume {
                let position = operators.volume_points[element * reference.n_volume + q];
                let weight = operators.volume_weights[element * reference.n_volume + q];
                let sample = field(&position)[component];
                for (i, slot) in load.iter_mut().enumerate() {
                    *slot += weight * reference.volume_values[q * reference.n_dof + i] * sample;
                }
            }
            for i in 0..reference.n_dof {
                let mut value = 0.0;
                for (j, slot) in load.iter().enumerate() {
                    value += reference.mass_inv[i * reference.n_dof + j] * slot;
                }
                values[base + i] = value / det_j;
            }
        }
    }
    if values.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "projected field",
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Solution sample at one volume quadrature point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadratureSample {
    /// Physical position `[m]`.
    pub position: [f64; 3],
    /// Physical quadrature weight.
    pub weight: f64,
    /// Pressure perturbation `[Pa]`.
    pub pressure: f64,
    /// Particle-velocity perturbation `[m/s]`.
    pub velocity: [f64; 3],
}

/// Sample the solver state at every volume quadrature point.
///
/// Powers cross-solver comparison gates: each sample carries its
/// physical position, weight, and evaluated state.
pub fn quadrature_samples(solver: &DgLeeSolver) -> Vec<QuadratureSample> {
    let operators = &solver.operators;
    let reference = &operators.reference;
    let dim = reference.dim;
    let mut samples = Vec::with_capacity(operators.elements.len() * reference.n_volume);
    for element in 0..operators.elements.len() {
        for q in 0..reference.n_volume {
            let mut pressure = 0.0;
            let mut velocity = [0.0; 3];
            for i in 0..reference.n_dof {
                let basis = reference.volume_values[q * reference.n_dof + i];
                pressure += solver.state.values[operators.component_base(0, element) + i] * basis;
                for (d, slot) in velocity.iter_mut().take(dim).enumerate() {
                    *slot +=
                        solver.state.values[operators.component_base(1 + d, element) + i] * basis;
                }
            }
            samples.push(QuadratureSample {
                position: operators.volume_points[element * reference.n_volume + q],
                weight: operators.volume_weights[element * reference.n_volume + q],
                pressure,
                velocity,
            });
        }
    }
    samples
}

/// Per-component L2 error against an exact solution.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for an unsupported mesh or a
/// mis-sized state.
pub fn l2_error(
    config: &DgLeeConfig,
    mesh: &Mesh,
    state: &DgLeeState,
    exact: ExteriorStateFn,
    time: f64,
) -> Result<Vec<f64>, AeroacousticsError> {
    let operators = build_operators(config.degree, mesh)?;
    let expected = operators.state_len();
    if state.values.len() != expected {
        return Err(AeroacousticsError::InvalidStateLength {
            expected,
            actual: state.values.len(),
        });
    }
    let reference = &operators.reference;
    let dim = reference.dim;
    let mut squared = vec![0.0; dim + 1];
    for element in 0..operators.elements.len() {
        for q in 0..reference.n_volume {
            let position = operators.volume_points[element * reference.n_volume + q];
            let weight = operators.volume_weights[element * reference.n_volume + q];
            let target = exact(&position, time);
            for component in 0..dim + 1 {
                let base = operators.component_base(component, element);
                let mut value = 0.0;
                for i in 0..reference.n_dof {
                    value +=
                        state.values[base + i] * reference.volume_values[q * reference.n_dof + i];
                }
                let difference = value - target[component];
                squared[component] += weight * difference * difference;
            }
        }
    }
    Ok(squared.iter().map(|error| error.sqrt()).collect())
}

/// Total acoustic energy of a DG perturbation `[J]`.
///
/// Integrates `p²/(2ρc²) + ρ|u|²/2` with the volume quadrature.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for an unsupported mesh or a
/// mis-sized state.
pub fn acoustic_energy_dg(
    config: &DgLeeConfig,
    mesh: &Mesh,
    state: &DgLeeState,
) -> Result<f64, AeroacousticsError> {
    let operators = build_operators(config.degree, mesh)?;
    let expected = operators.state_len();
    if state.values.len() != expected {
        return Err(AeroacousticsError::InvalidStateLength {
            expected,
            actual: state.values.len(),
        });
    }
    let reference = &operators.reference;
    let dim = reference.dim;
    let rho = config.density;
    let c = config.speed_of_sound;
    let mut total = 0.0;
    for element in 0..operators.elements.len() {
        for q in 0..reference.n_volume {
            let weight = operators.volume_weights[element * reference.n_volume + q];
            let pressure_base = operators.component_base(0, element);
            let mut pressure = 0.0;
            let mut kinetic = 0.0;
            for i in 0..reference.n_dof {
                pressure += state.values[pressure_base + i]
                    * reference.volume_values[q * reference.n_dof + i];
            }
            for d in 0..dim {
                let base = operators.component_base(1 + d, element);
                let mut velocity = 0.0;
                for i in 0..reference.n_dof {
                    velocity +=
                        state.values[base + i] * reference.volume_values[q * reference.n_dof + i];
                }
                kinetic += velocity * velocity;
            }
            total += weight * (pressure * pressure / (2.0 * rho * c * c) + rho * kinetic / 2.0);
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::super::drp_structured::{
        DrpStructuredConfig, DrpStructuredEuler, DrpStructuredState, StructuredAxis, StructuredFace,
    };
    use super::super::wave_diagnostics::group_velocity_from_track;
    use super::*;
    use crate::mesh::{box_mesh_tetrahedra, rectangular_mesh_triangles, unit_square_triangles};
    use std::f64::consts::PI;

    const RHO: f64 = 1.2;
    const C: f64 = 343.0;

    #[test]
    fn dense_inverse_is_accurate() {
        let matrix = [4.0, 1.0, 2.0, 1.0, 5.0, 0.5, 2.0, 0.5, 6.0];
        let inverse = invert_dense(&matrix, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut entry = 0.0;
                for k in 0..3 {
                    entry += matrix[i * 3 + k] * inverse[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((entry - expected).abs() < 1e-14, "entry[{i},{j}]={entry}");
            }
        }
    }

    #[test]
    fn reference_mass_inverse_is_accurate_for_all_supported_orders() {
        for element in [ElementType::Triangle, ElementType::Tetrahedron] {
            for degree in [
                PolynomialDegree::P1,
                PolynomialDegree::P2,
                PolynomialDegree::P3,
            ] {
                let dim = element.dimension();
                let space = DGSpace::new(degree);
                let admitted = AdmittedMesh {
                    dim,
                    element,
                    n_dof: space.num_local_dofs(element),
                };
                let reference = reference_operators(&admitted, degree);
                // Rebuild the mass matrix from cached tabulation.
                let n = admitted.n_dof;
                let mut mass = vec![0.0; n * n];
                for q in 0..reference.n_volume {
                    for i in 0..n {
                        for j in 0..n {
                            mass[i * n + j] += reference.volume_weights[q]
                                * reference.volume_values[q * n + i]
                                * reference.volume_values[q * n + j];
                        }
                    }
                }
                let mut worst = 0.0f64;
                for i in 0..n {
                    for j in 0..n {
                        let mut entry = 0.0;
                        for k in 0..n {
                            entry += mass[i * n + k] * reference.mass_inv[k * n + j];
                        }
                        let expected = if i == j { 1.0 } else { 0.0 };
                        worst = worst.max((entry - expected).abs());
                    }
                }
                assert!(
                    worst < 1e-10,
                    "{element:?} {degree:?}: inversion error={worst:e}"
                );
            }
        }
    }

    #[test]
    fn element_jacobian_transpose_handles_sheared_triangles() {
        // Shear the top half so Jacobians are non-symmetric: the
        // inverse-transpose then differs from the plain inverse.
        let mut mesh = unit_square_triangles(2);
        for node in mesh.nodes.iter_mut() {
            if node.y > 0.5 {
                node.x += 0.2;
            }
        }
        let admitted = admit_mesh(&mesh, PolynomialDegree::P1).unwrap();
        let mut sheared = 0;
        for elem_idx in 0..mesh.num_elements() {
            let data = element_data(&mesh, elem_idx, &admitted).unwrap();
            let vertices = mesh.elements[elem_idx].vertices();
            let origin = mesh.nodes[vertices[0]];
            let mut edges = [[0.0; 2]; 2];
            for (axis, edge) in edges.iter_mut().enumerate().take(2) {
                let node = mesh.nodes[vertices[axis + 1]];
                edge[0] = node.x - origin.x;
                edge[1] = node.y - origin.y;
            }
            if (edges[0][1] - edges[1][0]).abs() > 1e-12 {
                sheared += 1;
            }
            // inv_t · J^T = I with J[row][col] = edges[col][row].
            for i in 0..2 {
                for j in 0..2 {
                    let mut entry = 0.0;
                    for k in 0..2 {
                        entry += data.inv_jacobian_t[i][k] * edges[k][j];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (entry - expected).abs() < 1e-12,
                        "element {elem_idx}: entry[{i},{j}]={entry}"
                    );
                }
            }
        }
        assert!(sheared > 0, "the test mesh must contain sheared elements");
    }

    #[test]
    fn volume_quadrature_reproduces_domain_volumes() {
        for (mesh, expected) in [
            (unit_square_triangles(3), 1.0),
            (
                box_mesh_tetrahedra(0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 2, 1, 1),
                1.0,
            ),
        ] {
            let operators = build_operators(PolynomialDegree::P1, &mesh).unwrap();
            let total: f64 = operators.volume_weights.iter().sum();
            assert!((total - expected).abs() < 1e-12, "volume={total}");
        }
    }

    #[test]
    fn face_quadrature_reproduces_facet_areas() {
        let mesh = unit_square_triangles(3);
        let operators = build_operators(PolynomialDegree::P2, &mesh).unwrap();
        let mut checked = 0;
        for facet in build_interior_facets(&mesh) {
            let mut sorted = facet.shared_nodes.clone();
            sorted.sort_unstable();
            let area = facet_area_from_nodes(&sorted_corner_coords(&mesh, &sorted));
            let stored = operators
                .interior
                .iter()
                .find(|data| {
                    (data.plus.element == facet.elem_plus && data.minus.element == facet.elem_minus)
                        || (data.plus.element == facet.elem_minus
                            && data.minus.element == facet.elem_plus)
                })
                .unwrap();
            let total: f64 = stored.weights.iter().sum();
            assert!(
                (total - area).abs() < 1e-12,
                "interior area={area} vs {total}"
            );
            checked += 1;
        }
        for (face, stored) in mesh.boundaries.iter().zip(operators.boundary.iter()) {
            let mut sorted = face.nodes.clone();
            sorted.sort_unstable();
            let area = facet_area_from_nodes(&sorted_corner_coords(&mesh, &sorted));
            let total: f64 = stored.weights.iter().sum();
            assert!(
                (total - area).abs() < 1e-12,
                "boundary area={area} vs {total}"
            );
            checked += 1;
        }
        assert!(checked > 0);
    }

    /// Reference coordinates of one facet side at face quadrature.
    fn side_reference_coords(
        mesh: &Mesh,
        admitted: &AdmittedMesh,
        reference: &ReferenceOperators,
        elem_idx: usize,
        sorted: &[usize],
    ) -> Vec<[f64; 3]> {
        let local = local_face_indices(mesh, elem_idx, sorted);
        let ref_vertices = reference_vertices(admitted.element);
        let mut coords = Vec::with_capacity(reference.n_face);
        for q in 0..reference.n_face {
            let mut point = [0.0; 3];
            for (bary, local_idx) in reference.face_barycentric
                [q * reference.n_face_nodes..(q + 1) * reference.n_face_nodes]
                .iter()
                .zip(local.iter())
            {
                let vertex = ref_vertices[*local_idx];
                point[0] += bary * vertex[0];
                point[1] += bary * vertex[1];
                point[2] += bary * vertex[2];
            }
            coords.push(point);
        }
        coords
    }

    #[test]
    fn face_mapping_lands_on_physical_facets() {
        // Independent P1 re-mapping: reference face points must recover
        // the stored physical face quadrature on every side.
        for mesh in [
            unit_square_triangles(2),
            box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1),
        ] {
            let admitted = admit_mesh(&mesh, PolynomialDegree::P2).unwrap();
            let operators = build_operators(PolynomialDegree::P2, &mesh).unwrap();
            let mut sides = 0;
            let mut check_side = |elem_idx: usize, sorted: &[usize], stored: &[[f64; 3]]| {
                let vertices = mesh.elements[elem_idx].vertices();
                let corners: Vec<[f64; 3]> = vertices
                    .iter()
                    .map(|node| {
                        let point = mesh.nodes[*node];
                        [point.x, point.y, point.z]
                    })
                    .collect();
                let reference =
                    side_reference_coords(&mesh, &admitted, &operators.reference, elem_idx, sorted);
                for (coords, expected) in reference.iter().zip(stored.iter()) {
                    let mut mapped = corners[0];
                    for (axis, corner) in corners.iter().enumerate().skip(1).take(admitted.dim) {
                        mapped[0] += coords[axis - 1] * (corner[0] - corners[0][0]);
                        mapped[1] += coords[axis - 1] * (corner[1] - corners[0][1]);
                        mapped[2] += coords[axis - 1] * (corner[2] - corners[0][2]);
                    }
                    let error = ((mapped[0] - expected[0]).powi(2)
                        + (mapped[1] - expected[1]).powi(2)
                        + (mapped[2] - expected[2]).powi(2))
                    .sqrt();
                    assert!(error < 1e-12, "mapping error={error:e}");
                }
                sides += 1;
            };
            for facet in build_interior_facets(&mesh) {
                let mut sorted = facet.shared_nodes.clone();
                sorted.sort_unstable();
                let expected =
                    face_points(&sorted_corner_coords(&mesh, &sorted), &operators.reference);
                check_side(facet.elem_plus, &sorted, &expected);
                check_side(facet.elem_minus, &sorted, &expected);
            }
            for (face, stored) in mesh.boundaries.iter().zip(operators.boundary.iter()) {
                let mut sorted = face.nodes.clone();
                sorted.sort_unstable();
                check_side(face.element_idx, &sorted, &stored.points);
            }
            assert!(sides > 0);
        }
    }

    #[test]
    fn facet_normals_point_plus_to_minus_and_outward() {
        let mesh = unit_square_triangles(2);
        let operators = build_operators(PolynomialDegree::P1, &mesh).unwrap();
        for facet in &operators.interior {
            let from = element_centroid(&mesh, facet.plus.element);
            let to = element_centroid(&mesh, facet.minus.element);
            let along = (to[0] - from[0]) * facet.normal[0]
                + (to[1] - from[1]) * facet.normal[1]
                + (to[2] - from[2]) * facet.normal[2];
            assert!(along > 0.0, "interior normal misoriented");
        }
        for (face, facet) in mesh.boundaries.iter().zip(operators.boundary.iter()) {
            let centroid = element_centroid(&mesh, face.element_idx);
            // Outward check: the normal points from the element centroid
            // to the face midpoint.
            let face_mid = [
                facet.points.iter().map(|point| point[0]).sum::<f64>() / facet.points.len() as f64,
                facet.points.iter().map(|point| point[1]).sum::<f64>() / facet.points.len() as f64,
                facet.points.iter().map(|point| point[2]).sum::<f64>() / facet.points.len() as f64,
            ];
            let along = (face_mid[0] - centroid[0]) * facet.normal[0]
                + (face_mid[1] - centroid[1]) * facet.normal[1]
                + (face_mid[2] - centroid[2]) * facet.normal[2];
            assert!(along > 0.0, "boundary normal misoriented");
        }
    }

    #[test]
    fn upwind_flux_is_consistent_and_fully_upwind_when_supersonic() {
        let flow = [30.0, -10.0, 5.0];
        // Normals must lie in the velocity space (the solver only
        // passes in-plane unit facet normals).
        for (normal, dim) in [
            ([1.0f64, 0.0, 0.0], 2),
            ([0.0, 1.0, 0.0], 2),
            ([0.6, 0.8, 0.0], 2),
            ([1.0, 0.0, 0.0], 3),
            ([0.0, 1.0, 0.0], 3),
            ([0.0, 0.0, 1.0], 3),
            ([0.6, 0.8, 0.0], 3),
            (
                [
                    -0.577_350_269_189_625_8,
                    -0.577_350_269_189_625_8,
                    -0.577_350_269_189_625_8,
                ],
                3,
            ),
        ] {
            let norm = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
            let normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
            {
                let state = [1.5, 0.4, -0.7, 0.2];
                let mut starred = [0.0; 4];
                let mut physical = [0.0; 4];
                upwind_flux(&state, &state, &normal, dim, &flow, RHO, C, &mut starred);
                physical_flux_normal(&state, &normal, dim, &flow, RHO, C, &mut physical);
                for component in 0..dim + 1 {
                    // Relative gate: O(1e5) flux magnitudes leave O(1e-11)
                    // cancellation roundoff in the dissipation assembly.
                    let scale = physical[component].abs().max(1.0);
                    assert!(
                        (starred[component] - physical[component]).abs() < 1e-12 * scale,
                        "consistency violated"
                    );
                }
                // Supersonic outflow: every characteristic leaves, so the
                // flux equals the interior physical flux exactly.
                let fast = [500.0 * normal[0], 500.0 * normal[1], 500.0 * normal[2]];
                let exterior = [0.5, -1.1, 0.3, 0.9];
                upwind_flux(&state, &exterior, &normal, dim, &fast, RHO, C, &mut starred);
                physical_flux_normal(&state, &normal, dim, &fast, RHO, C, &mut physical);
                for component in 0..dim + 1 {
                    let scale = physical[component].abs().max(1.0);
                    assert!(
                        (starred[component] - physical[component]).abs() < 1e-12 * scale,
                        "supersonic upwinding violated"
                    );
                }
            }
        }
    }

    #[test]
    fn config_validation_rejects_bad_inputs() {
        let mesh = unit_square_triangles(2);
        let quads = crate::mesh::rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, 2, 2);
        assert!(matches!(
            DgLeeConfig::with_cfl(PolynomialDegree::P1, RHO, C, [0.0, 0.0, 0.0], &quads, 0.5),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
        assert!(matches!(
            DgLeeConfig::with_cfl(PolynomialDegree::P1, RHO, C, [0.0, 0.0, 1.0], &mesh, 0.5),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
        assert!(matches!(
            DgLeeConfig::with_cfl(PolynomialDegree::P1, RHO, C, [0.0, 0.0, 0.0], &mesh, 1.5),
            Err(AeroacousticsError::InvalidCfl(_))
        ));
        let config =
            DgLeeConfig::with_cfl(PolynomialDegree::P1, RHO, C, [10.0, 0.0, 0.0], &mesh, 0.5)
                .unwrap();
        let mut over_cfl = config.clone();
        over_cfl.time_step *= 2.1;
        assert!(matches!(
            over_cfl.validate(&mesh),
            Err(AeroacousticsError::InvalidCfl(_))
        ));
        // Flipping one triangle inverts its Jacobian.
        let mut flipped = unit_square_triangles(1);
        flipped.elements[0].nodes.swap(0, 1);
        assert!(matches!(
            DgLeeConfig::with_cfl(PolynomialDegree::P1, RHO, C, [0.0, 0.0, 0.0], &flipped, 0.5),
            Err(AeroacousticsError::DegenerateElement { index: 0 })
        ));
        // Mis-sized and non-finite states are rejected at construction.
        let state = DgLeeState::zeros(&config, &mesh).unwrap();
        let short = DgLeeState {
            values: vec![0.0; state.values.len() - 1],
            ..state.clone()
        };
        assert!(matches!(
            DgLeeSolver::new(config.clone(), &mesh, short),
            Err(AeroacousticsError::InvalidStateLength { .. })
        ));
        let mut dirty = state;
        dirty.values[3] = f64::NAN;
        assert!(matches!(
            DgLeeSolver::new(config, &mesh, dirty),
            Err(AeroacousticsError::NonFiniteField { .. })
        ));
    }

    // ------------------------------------------------------------------
    // Manufactured solutions
    // ------------------------------------------------------------------

    const MMS_UX: f64 = 30.0;
    const MMS_UY: f64 = -20.0;
    const MMS_UZ: f64 = 10.0;
    const MMS_OMEGA: f64 = 3.0;

    fn mms_2d_exact(position: &[f64; 3], time: f64) -> [f64; 4] {
        let (s, c) = (2.0 * PI * position[0]).sin_cos();
        let (sy, cy) = (2.0 * PI * position[1]).sin_cos();
        let (st, ct) = (MMS_OMEGA * time).sin_cos();
        [s * sy * ct, c * sy * st, s * cy * st, 0.0]
    }

    fn mms_2d_source(position: &[f64; 3], time: f64) -> [f64; 4] {
        let k = 2.0 * PI;
        let (s, c) = (k * position[0]).sin_cos();
        let (sy, cy) = (k * position[1]).sin_cos();
        let (st, ct) = (MMS_OMEGA * time).sin_cos();
        let dt_p = -MMS_OMEGA * s * sy * st;
        let dx_p = k * c * sy * ct;
        let dy_p = k * s * cy * ct;
        let dt_ux = MMS_OMEGA * c * sy * ct;
        let dx_ux = -k * s * sy * st;
        let dy_ux = k * c * cy * st;
        let dt_uy = MMS_OMEGA * s * cy * ct;
        let dx_uy = k * c * cy * st;
        let dy_uy = -k * s * sy * st;
        [
            dt_p + MMS_UX * dx_p + MMS_UY * dy_p + RHO * C * C * (dx_ux + dy_uy),
            dt_ux + MMS_UX * dx_ux + MMS_UY * dy_ux + dx_p / RHO,
            dt_uy + MMS_UX * dx_uy + MMS_UY * dy_uy + dy_p / RHO,
            0.0,
        ]
    }

    fn mms_3d_exact(position: &[f64; 3], time: f64) -> [f64; 4] {
        let (s, c) = (2.0 * PI * position[0]).sin_cos();
        let (sy, cy) = (2.0 * PI * position[1]).sin_cos();
        let (sz, cz) = (2.0 * PI * position[2]).sin_cos();
        let (st, ct) = (MMS_OMEGA * time).sin_cos();
        [
            s * sy * sz * ct,
            c * sy * sz * st,
            s * cy * sz * st,
            s * sy * cz * st,
        ]
    }

    fn mms_3d_source(position: &[f64; 3], time: f64) -> [f64; 4] {
        let k = 2.0 * PI;
        let (s, c) = (k * position[0]).sin_cos();
        let (sy, cy) = (k * position[1]).sin_cos();
        let (sz, cz) = (k * position[2]).sin_cos();
        let (st, ct) = (MMS_OMEGA * time).sin_cos();
        let dt_p = -MMS_OMEGA * s * sy * sz * st;
        let dx_p = k * c * sy * sz * ct;
        let dy_p = k * s * cy * sz * ct;
        let dz_p = k * s * sy * cz * ct;
        let dt_ux = MMS_OMEGA * c * sy * sz * ct;
        let dx_ux = -k * s * sy * sz * st;
        let dy_ux = k * c * cy * sz * st;
        let dz_ux = k * c * sy * cz * st;
        let dt_uy = MMS_OMEGA * s * cy * sz * ct;
        let dx_uy = k * c * cy * sz * st;
        let dy_uy = -k * s * sy * sz * st;
        let dz_uy = k * s * cy * cz * st;
        let dt_uz = MMS_OMEGA * s * sy * cz * ct;
        let dx_uz = k * c * sy * cz * st;
        let dy_uz = k * s * cy * cz * st;
        let dz_uz = -k * s * sy * sz * st;
        let convect = |dx: f64, dy: f64, dz: f64| MMS_UX * dx + MMS_UY * dy + MMS_UZ * dz;
        [
            dt_p + convect(dx_p, dy_p, dz_p) + RHO * C * C * (dx_ux + dy_uy + dz_uz),
            dt_ux + convect(dx_ux, dy_ux, dz_ux) + dx_p / RHO,
            dt_uy + convect(dx_uy, dy_uy, dz_uy) + dy_p / RHO,
            dt_uz + convect(dx_uz, dy_uz, dz_uz) + dz_p / RHO,
        ]
    }

    /// Prescribe the exact state on every boundary marker the mesh carries
    /// (structured triangle meshes tag sides 1-4; tet meshes use 0).
    fn prescribe_all(config: DgLeeConfig, mesh: &Mesh, exterior: ExteriorStateFn) -> DgLeeConfig {
        let mut markers: Vec<i32> = mesh.boundaries.iter().map(|face| face.marker).collect();
        markers.sort_unstable();
        markers.dedup();
        let mut configured = config;
        for marker in markers {
            configured = configured.with_boundary(marker, DgLeeBoundary::Prescribed(exterior));
        }
        configured
    }

    fn mms_error_2d(n: usize, degree: PolynomialDegree, end: f64) -> Vec<f64> {
        let mesh = unit_square_triangles(n);
        let config = prescribe_all(
            DgLeeConfig::with_cfl(degree, RHO, C, [MMS_UX, MMS_UY, 0.0], &mesh, 0.5).unwrap(),
            &mesh,
            mms_2d_exact,
        );
        let init = DgLeeState::project(&config, &mesh, |point| mms_2d_exact(point, 0.0)).unwrap();
        let mut solver = DgLeeSolver::new(config.clone(), &mesh, init).unwrap();
        solver.source = Some(mms_2d_source);
        solver.advance((end / solver.config.time_step).ceil() as usize);
        l2_error(&config, &mesh, &solver.state, mms_2d_exact, solver.time).unwrap()
    }

    fn mms_error_3d(n: usize, end: f64) -> Vec<f64> {
        let mesh = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, n, n);
        let config = DgLeeConfig::with_cfl(
            PolynomialDegree::P1,
            RHO,
            C,
            [MMS_UX, MMS_UY, MMS_UZ],
            &mesh,
            0.5,
        )
        .unwrap()
        .with_boundary(0, DgLeeBoundary::Prescribed(mms_3d_exact));
        let init = DgLeeState::project(&config, &mesh, |point| mms_3d_exact(point, 0.0)).unwrap();
        let mut solver = DgLeeSolver::new(config.clone(), &mesh, init).unwrap();
        solver.source = Some(mms_3d_source);
        solver.advance((end / solver.config.time_step).ceil() as usize);
        l2_error(&config, &mesh, &solver.state, mms_3d_exact, solver.time).unwrap()
    }

    fn convergence_rate(coarse: f64, fine: f64, ratio: f64) -> f64 {
        (coarse / fine).ln() / ratio.ln()
    }

    #[test]
    fn manufactured_solution_converges_at_degree_plus_one_2d() {
        // P1 starts at n = 4 (k*h ~ 1, pre-asymptotic), so its coarse
        // rate gate is looser; the refined rate must still approach 2.
        for (degree, levels, minimum) in [
            (PolynomialDegree::P1, [4, 8, 16], 1.6),
            (PolynomialDegree::P2, [4, 8, 16], 2.7),
            (PolynomialDegree::P3, [4, 8, 16], 3.4),
        ] {
            let errors: Vec<Vec<f64>> = levels
                .iter()
                .map(|n| mms_error_2d(*n, degree, 0.005))
                .collect();
            for component in 0..3 {
                assert!(
                    errors[0][component] > errors[1][component]
                        && errors[1][component] > errors[2][component],
                    "{degree:?} component {component}: errors not monotone: {:?}",
                    errors
                        .iter()
                        .map(|error| error[component])
                        .collect::<Vec<_>>()
                );
                let rate1 = convergence_rate(errors[0][component], errors[1][component], 2.0);
                let rate2 = convergence_rate(errors[1][component], errors[2][component], 2.0);
                assert!(
                    rate1 > minimum && rate2 > minimum,
                    "{degree:?} component {component}: rates {rate1:.2}/{rate2:.2} below {minimum}"
                );
            }
        }
    }

    #[test]
    fn manufactured_solution_converges_on_tetrahedra() {
        let errors: Vec<Vec<f64>> = [3, 4, 5].iter().map(|n| mms_error_3d(*n, 0.005)).collect();
        for component in 0..4 {
            let rate1 = convergence_rate(errors[0][component], errors[1][component], 4.0 / 3.0);
            let rate2 = convergence_rate(errors[1][component], errors[2][component], 5.0 / 4.0);
            assert!(
                rate1 > 1.5 && rate2 > 1.5,
                "P1 tet component {component}: rates {rate1:.2}/{rate2:.2}"
            );
        }
    }

    fn linear_exterior(position: &[f64; 3], _time: f64) -> [f64; 4] {
        [position[0], 0.0, 0.0, 0.0]
    }

    #[test]
    fn linear_field_rhs_is_exact() {
        // p = x, u = 0, U = (30, -20): exact rates are (-30, -1/rho, 0).
        // Linear fields have zero jumps once the exact exterior is
        // prescribed, so this isolates the volume divergence path.
        let mesh = unit_square_triangles(2);
        let config = prescribe_all(
            DgLeeConfig::with_cfl(PolynomialDegree::P1, RHO, C, [30.0, -20.0, 0.0], &mesh, 0.5)
                .unwrap(),
            &mesh,
            linear_exterior,
        );
        let state = DgLeeState::project(&config, &mesh, |point| [point[0], 0.0, 0.0, 0.0]).unwrap();
        let operators = build_operators(PolynomialDegree::P1, &mesh).unwrap();
        let mut load = vec![0.0; state.values.len()];
        let mut deriv = vec![0.0; state.values.len()];
        rhs(
            &operators,
            &config,
            &state.values,
            0.0,
            None,
            &mut load,
            &mut deriv,
        );
        let n = operators.admitted.n_dof * mesh.num_elements();
        let mut worst = [0.0f64; 3];
        for i in 0..n {
            worst[0] = worst[0].max((deriv[i] + 30.0).abs());
            worst[1] = worst[1].max((deriv[n + i] + 1.0 / RHO).abs());
            worst[2] = worst[2].max(deriv[2 * n + i].abs());
        }
        assert!(
            worst[0] < 1e-11 && worst[1] < 1e-11 && worst[2] < 1e-11,
            "linear RHS deviations: {worst:?}"
        );
    }

    fn constant_2d_exterior(_position: &[f64; 3], _time: f64) -> [f64; 4] {
        [1.5, 0.3, -0.2, 0.0]
    }

    fn constant_3d_exterior(_position: &[f64; 3], _time: f64) -> [f64; 4] {
        [1.5, 0.3, -0.2, 0.1]
    }

    #[test]
    fn constant_state_is_preserved_to_roundoff() {
        // Uniform states annihilate every term (zero divergence, zero
        // jumps), so one step must reproduce them to roundoff.
        for degree in [
            PolynomialDegree::P1,
            PolynomialDegree::P2,
            PolynomialDegree::P3,
        ] {
            let mesh = unit_square_triangles(4);
            let config = prescribe_all(
                DgLeeConfig::with_cfl(degree, RHO, C, [MMS_UX, MMS_UY, 0.0], &mesh, 0.5).unwrap(),
                &mesh,
                constant_2d_exterior,
            );
            let init = DgLeeState::project(&config, &mesh, |_| [1.5, 0.3, -0.2, 0.0]).unwrap();
            let mut solver = DgLeeSolver::new(config, &mesh, init).unwrap();
            let before = solver.state.values.clone();
            solver.step();
            let drift = solver
                .state
                .values
                .iter()
                .zip(before.iter())
                .map(|(after, prior)| (after - prior).abs())
                .fold(0.0f64, f64::max);
            assert!(drift < 1e-11, "{degree:?}: drift={drift:e}");
        }
        let mesh = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2);
        let config = prescribe_all(
            DgLeeConfig::with_cfl(
                PolynomialDegree::P1,
                RHO,
                C,
                [MMS_UX, MMS_UY, MMS_UZ],
                &mesh,
                0.5,
            )
            .unwrap(),
            &mesh,
            constant_3d_exterior,
        );
        let init = DgLeeState::project(&config, &mesh, |_| [1.5, 0.3, -0.2, 0.1]).unwrap();
        let mut solver = DgLeeSolver::new(config, &mesh, init).unwrap();
        let before = solver.state.values.clone();
        solver.step();
        let drift = solver
            .state
            .values
            .iter()
            .zip(before.iter())
            .map(|(after, prior)| (after - prior).abs())
            .fold(0.0f64, f64::max);
        assert!(drift < 1e-11, "tet drift={drift:e}");
    }

    // ------------------------------------------------------------------
    // Propagation gates
    // ------------------------------------------------------------------

    /// Plane packet field uniform in y, polarized downstream along x.
    fn plane_packet_field(
        center: f64,
        sigma: f64,
        wavenumber: f64,
    ) -> impl Fn(&[f64; 3]) -> [f64; 4] {
        move |point: &[f64; 3]| {
            let delta = point[0] - center;
            let envelope = (-0.5 * (delta / sigma).powi(2)).exp();
            let pressure = envelope * (wavenumber * delta).cos();
            [pressure, pressure / (RHO * C), 0.0, 0.0]
        }
    }

    /// Energy centroid of DG samples along x.
    fn sample_centroid(samples: &[QuadratureSample]) -> Option<f64> {
        let mut moment = 0.0;
        let mut total = 0.0;
        for sample in samples {
            let weight = sample.weight * sample.pressure.powi(2);
            moment += sample.position[0] * weight;
            total += weight;
        }
        if total == 0.0 {
            None
        } else {
            Some(moment / total)
        }
    }

    #[test]
    fn packet_group_velocity_matches_analytic_speed() {
        let mesh = rectangular_mesh_triangles(0.0, 8.0, 0.0, 1.0, 64, 8);
        // Slip walls top/bottom (markers 3, 4): a y-uniform plane wave
        // satisfies u_y = 0 exactly, so walls are non-intrusive, while
        // zero-correction outflow at grazing incidence would inject
        // spurious transverse waves that lag the packet and bias the
        // energy centroid.
        let config =
            DgLeeConfig::with_cfl(PolynomialDegree::P2, RHO, C, [20.0, 0.0, 0.0], &mesh, 0.5)
                .unwrap()
                .with_boundary(3, DgLeeBoundary::SlipWall)
                .with_boundary(4, DgLeeBoundary::SlipWall);
        let field = plane_packet_field(2.0, 0.5, 8.0);
        let init = DgLeeState::project(&config, &mesh, |point| field(point)).unwrap();
        let mut solver = DgLeeSolver::new(config.clone(), &mesh, init).unwrap();
        let mut times = Vec::new();
        let mut centroids = Vec::new();
        for _ in 0..14 {
            solver.advance(80);
            let centroid = sample_centroid(&quadrature_samples(&solver)).unwrap();
            if centroid > 6.0 {
                break;
            }
            times.push(solver.time);
            centroids.push(centroid);
        }
        assert!(times.len() >= 6, "track too short: {}", times.len());
        let measured = group_velocity_from_track(&times, &centroids).unwrap();
        let analytic = 20.0 + C;
        assert!(
            (measured - analytic).abs() / analytic < 2e-3,
            "measured={measured}, analytic={analytic}"
        );
    }

    #[test]
    fn slip_wall_reflects_packet_upstream() {
        let mut mesh = rectangular_mesh_triangles(0.0, 3.0, 0.0, 0.5, 48, 8);
        for face in mesh.boundaries.iter_mut() {
            let centroid_x = face
                .nodes
                .iter()
                .map(|node| mesh.nodes[*node].x)
                .sum::<f64>()
                / face.nodes.len() as f64;
            if centroid_x > 2.99 {
                face.marker = 1;
            }
        }
        assert!(mesh.boundaries.iter().any(|face| face.marker == 1));
        let config =
            DgLeeConfig::with_cfl(PolynomialDegree::P1, RHO, C, [20.0, 0.0, 0.0], &mesh, 0.8)
                .unwrap()
                .with_boundary(1, DgLeeBoundary::SlipWall)
                .with_boundary(3, DgLeeBoundary::SlipWall)
                .with_boundary(4, DgLeeBoundary::SlipWall);
        let field = plane_packet_field(0.75, 0.25, 8.0);
        let init = DgLeeState::project(&config, &mesh, |point| field(point)).unwrap();
        let mut solver = DgLeeSolver::new(config.clone(), &mesh, init).unwrap();
        let initial = acoustic_energy_dg(&config, &mesh, &solver.state).unwrap();
        // The packet reaches the wall at ~6.2 ms; fit the reflected leg.
        let mut times = Vec::new();
        let mut centroids = Vec::new();
        while solver.time < 0.010 {
            solver.advance(40);
            if solver.time > 0.007 {
                let centroid = sample_centroid(&quadrature_samples(&solver)).unwrap();
                times.push(solver.time);
                centroids.push(centroid);
            }
        }
        assert!(times.len() >= 4, "track too short: {}", times.len());
        let measured = group_velocity_from_track(&times, &centroids).unwrap();
        let analytic = 20.0 - C;
        assert!(
            (measured - analytic).abs() / analytic.abs() < 3e-2,
            "measured={measured}, analytic={analytic}"
        );
        // A passive wall cannot create energy.
        let remaining = acoustic_energy_dg(&config, &mesh, &solver.state).unwrap() / initial;
        assert!(remaining < 1.05, "energy growth={remaining}");
        assert!(
            solver.state.values.iter().all(|x| x.is_finite()),
            "non-finite state after reflection"
        );
    }

    // ------------------------------------------------------------------
    // Structured/DG equivalence gate
    // ------------------------------------------------------------------

    /// Bilinear sample of a periodic structured pressure field.
    fn sample_structured(
        config: &DrpStructuredConfig,
        state: &DrpStructuredState,
        x: f64,
        y: f64,
    ) -> f64 {
        let nx = config.axes[0].n_points;
        let ny = config.axes[1].n_points;
        let fx = (x / config.dx(0)).rem_euclid(nx as f64);
        let fy = (y / config.dx(1)).rem_euclid(ny as f64);
        let ix = fx.floor() as usize % nx;
        let iy = fy.floor() as usize % ny;
        let jx = (ix + 1) % nx;
        let jy = (iy + 1) % ny;
        let tx = fx - fx.floor();
        let ty = fy - fy.floor();
        let at = |a: usize, b: usize| state.pressure[a + nx * b];
        (1.0 - tx) * (1.0 - ty) * at(ix, iy)
            + tx * (1.0 - ty) * at(jx, iy)
            + (1.0 - tx) * ty * at(ix, jy)
            + tx * ty * at(jx, jy)
    }

    #[test]
    fn structured_and_dg_solvers_agree_on_a_shared_packet() {
        // Identical plane packet, identical end time: the structured step
        // is set to the DG step (valid under both CFL contracts), so both
        // solvers land on the same instant with no time interpolation.
        // X-refined (the envelope/carrier vary only in x): h_x = 0.0625
        // keeps the DG P2 projection error near 1e-3.
        let dg_mesh = rectangular_mesh_triangles(0.0, 4.0, 0.0, 4.0, 64, 32);
        // Slip walls top/bottom (markers 3, 4) preserve the y-uniform
        // plane packet exactly, matching the structured periodic sides.
        let dg_config = DgLeeConfig::with_cfl(
            PolynomialDegree::P2,
            RHO,
            C,
            [20.0, 0.0, 0.0],
            &dg_mesh,
            0.5,
        )
        .unwrap()
        .with_boundary(3, DgLeeBoundary::SlipWall)
        .with_boundary(4, DgLeeBoundary::SlipWall);
        let field = plane_packet_field(1.5, 0.4, 6.0);
        let dg_init = DgLeeState::project(&dg_config, &dg_mesh, |point| field(point)).unwrap();
        let mut dg = DgLeeSolver::new(dg_config.clone(), &dg_mesh, dg_init).unwrap();

        // Fine structured grid (dx = 0.02): bilinear sampling at DG
        // quadrature points then contributes ~1e-3 interpolation error.
        let axis = StructuredAxis {
            n_points: 200,
            length: 4.0,
            lower: StructuredFace::Periodic,
            upper: StructuredFace::Periodic,
        };
        let mut structured_config =
            DrpStructuredConfig::with_cfl(vec![axis, axis], RHO, C, vec![20.0, 0.0], 0.5).unwrap();
        structured_config.time_step = dg_config.time_step;
        structured_config.validate().unwrap();
        // Same centered carrier as plane_packet_field (cos(k*(x - 1.5))),
        // not the plane_wave helper whose phase vanishes at x = 0.
        let mut structured_state = DrpStructuredState::zeros(&structured_config);
        for node in 0..structured_config.n_nodes() {
            let index = node % structured_config.axes[0].n_points;
            let x = index as f64 * structured_config.dx(0);
            let delta = x - 1.5;
            let envelope = (-0.5 * (delta / 0.4).powi(2)).exp();
            let pressure = envelope * (6.0 * delta).cos();
            structured_state.pressure[node] = pressure;
            structured_state.velocity[0][node] = pressure / (RHO * C);
        }
        let mut structured =
            DrpStructuredEuler::new(structured_config.clone(), structured_state).unwrap();

        let steps = (0.002_2 / dg_config.time_step).round() as usize;
        dg.advance(steps);
        structured.advance(steps);
        assert_eq!(dg.time, structured.time);

        // Sampled L2 difference at DG quadrature points.
        let samples = quadrature_samples(&dg);
        let mut difference = 0.0;
        let mut energy = 0.0;
        let mut dg_moment = 0.0;
        let mut dg_total = 0.0;
        let mut structured_moment = 0.0;
        let mut structured_total = 0.0;
        for sample in &samples {
            let reference = sample_structured(
                &structured_config,
                &structured.state,
                sample.position[0],
                sample.position[1],
            );
            difference += sample.weight * (sample.pressure - reference).powi(2);
            energy += sample.weight * sample.pressure.powi(2);
            dg_moment += sample.position[0] * sample.weight * sample.pressure.powi(2);
            dg_total += sample.weight * sample.pressure.powi(2);
            structured_moment += sample.position[0] * sample.weight * reference.powi(2);
            structured_total += sample.weight * reference.powi(2);
        }
        let relative = (difference / energy).sqrt();
        assert!(relative < 5e-3, "sampled L2 difference={relative:e}");
        let centroid_gap = (dg_moment / dg_total - structured_moment / structured_total).abs();
        assert!(centroid_gap < 5e-3, "centroid gap={centroid_gap:e}");
    }
}
