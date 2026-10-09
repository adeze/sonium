//! Core function space trait and associated types

use crate::mesh::{ElementType, Mesh};

use super::contract::PiolaError;

/// Whether a function space produces scalar or vector-valued fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// Scalar field (pressure, temperature): 1 component per DOF
    Scalar,
    /// Vector field (displacement, E-field): `spatial_dim` components per DOF
    Vector,
}

/// Failure in a function-space topology/DOF contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpaceValidationError {
    /// The mesh connectivity is too short or references a missing node.
    InvalidMesh {
        element: usize,
        reason: &'static str,
    },
    /// An element returned the wrong number of local DOFs.
    LocalDofCount {
        element: usize,
        expected: usize,
        actual: usize,
    },
    /// A local-to-global map references a nonexistent global DOF.
    GlobalDofOutOfBounds {
        element: usize,
        dof: usize,
        num_global_dofs: usize,
    },
    /// A local map reuses a global DOF where the element basis requires unique
    /// local support points.
    DuplicateLocalDof { element: usize, dof: usize },
    /// The requested space order is not implemented by the space.
    UnsupportedOrder {
        requested: usize,
        supported: &'static str,
    },
    /// The space order is not implemented for this element family.
    UnsupportedElement {
        element: usize,
        element_type: ElementType,
        order: usize,
    },
}

impl std::fmt::Display for SpaceValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMesh { element, reason } => {
                write!(f, "invalid mesh element {element}: {reason}")
            }
            Self::LocalDofCount {
                element,
                expected,
                actual,
            } => write!(
                f,
                "element {element} has {actual} DOFs, expected {expected}"
            ),
            Self::GlobalDofOutOfBounds {
                element,
                dof,
                num_global_dofs,
            } => write!(
                f,
                "element {element} maps DOF {dof}, outside 0..{num_global_dofs}"
            ),
            Self::DuplicateLocalDof { element, dof } => {
                write!(
                    f,
                    "element {element} maps local support DOF {dof} more than once"
                )
            }
            Self::UnsupportedOrder {
                requested,
                supported,
            } => write!(
                f,
                "function-space order {requested} is unsupported; supported orders: {supported}"
            ),
            Self::UnsupportedElement {
                element,
                element_type,
                order,
            } => write!(
                f,
                "element {element} ({element_type:?}) does not support function-space order {order}"
            ),
        }
    }
}

impl std::error::Error for SpaceValidationError {}

/// Failure while checked-tabulating a function-space basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TabulationError {
    /// Mesh/topology admission failed before basis evaluation.
    Space(SpaceValidationError),
    /// A pointwise Piola map could not be evaluated safely.
    Piola(PiolaError),
}

impl std::fmt::Display for TabulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Space(error) => write!(f, "space tabulation rejected mesh: {error}"),
            Self::Piola(error) => write!(f, "space tabulation rejected Piola map: {error}"),
        }
    }
}

impl std::error::Error for TabulationError {}

impl From<SpaceValidationError> for TabulationError {
    fn from(error: SpaceValidationError) -> Self {
        Self::Space(error)
    }
}

impl From<PiolaError> for TabulationError {
    fn from(error: PiolaError) -> Self {
        Self::Piola(error)
    }
}

/// Tabulated basis function data at a single quadrature point.
///
/// Layout depends on the function space:
///
/// **Scalar H1** (`FieldType::Scalar`):
/// - `values`: `[N_0, N_1, ..., N_{n-1}]` (length = `n_dofs`)
/// - `derivatives`: physical-space gradients, flat `[dN_0/dx, dN_0/dy, ..., dN_1/dx, ...]`
///   (length = `n_dofs * spatial_dim`)
///
/// **Vector H(curl)** (`FieldType::Vector`):
/// - `values`: `[N_0x, N_0y, N_0z, N_1x, ...]` (length = `n_dofs * spatial_dim`)
/// - `derivatives`: physical-space curls, flat `[curl_N_0x, curl_N_0y, curl_N_0z, ...]`
///   (length = `n_dofs * curl_dim`, where curl_dim = 3 in 3D, 1 in 2D)
///
/// **H(div) RT0/RT1** (`FieldType::Vector`):
/// - `values`: `[N_0x, N_0y, N_0z, N_1x, ...]` (length = `n_dofs * spatial_dim`)
/// - `derivatives`: physical-space divergences `[div_N_0, div_N_1, ...]`
///   (length = `n_dofs`)
#[derive(Debug, Clone)]
pub struct BasisTabulation {
    /// Number of local DOFs
    pub n_dofs: usize,
    /// Spatial dimension (2 or 3)
    pub spatial_dim: usize,
    /// Basis function values (see struct docs for layout)
    pub values: Vec<f64>,
    /// Derivatives in physical space: gradients (H1) or curls (H(curl))
    pub derivatives: Vec<f64>,
    /// Physical-space Hessians for scalar H1 bases, flat by DOF then row-major
    /// `spatial_dim x spatial_dim`. Empty for spaces/elements without Hessian
    /// tabulation.
    pub hessians: Vec<f64>,
    /// Absolute value of Jacobian determinant
    pub det_j: f64,
    /// Physical coordinates of this quadrature point
    pub physical_coords: [f64; 3],
}

impl BasisTabulation {
    /// Get scalar basis function value for DOF `i` (H1 only).
    #[inline]
    pub fn value(&self, i: usize) -> f64 {
        self.values[i]
    }

    /// Get physical gradient component `d` of scalar DOF `i` (H1 only).
    #[inline]
    pub fn grad(&self, i: usize, d: usize) -> f64 {
        self.derivatives[i * self.spatial_dim + d]
    }

    /// Get vector basis function component `c` for DOF `i` (H(curl)/vector H1).
    #[inline]
    pub fn vector_value(&self, i: usize, c: usize) -> f64 {
        self.values[i * self.spatial_dim + c]
    }

    /// Get curl component `c` of vector DOF `i` (H(curl) only, 3D).
    #[inline]
    pub fn curl(&self, i: usize, c: usize) -> f64 {
        self.derivatives[i * 3 + c]
    }

    /// Get the physical divergence of H(div) basis function `i`.
    #[inline]
    pub fn divergence(&self, i: usize) -> f64 {
        debug_assert_eq!(
            self.derivatives.len(),
            self.n_dofs,
            "divergence is stored as one scalar per H(div) basis function"
        );
        self.derivatives[i]
    }

    /// Get physical Hessian entry `(row, col)` for scalar DOF `i`.
    #[inline]
    pub fn hessian(&self, i: usize, row: usize, col: usize) -> f64 {
        self.hessians[i * self.spatial_dim * self.spatial_dim + row * self.spatial_dim + col]
    }

    /// Whether this tabulation contains Hessian data.
    #[inline]
    pub fn has_hessians(&self) -> bool {
        !self.hessians.is_empty()
    }
}

/// A function space defines how DOFs are distributed on a mesh and how
/// basis functions are evaluated.
///
/// Implementations:
/// - [`H1Space`](super::H1Space): scalar Lagrange nodal elements (H1-conforming)
/// - [`VectorH1Space`](super::VectorH1Space): vector-valued Lagrange (for elasticity)
/// - `HcurlSpace`: Nedelec edge elements (H(curl)-conforming, for Maxwell) — future
pub trait FunctionSpace: Send + Sync {
    /// Whether this space produces scalar or vector fields.
    fn field_type(&self) -> FieldType;

    /// Number of components per evaluation point.
    /// - Scalar (H1): 1
    /// - Vector (H(curl), vector H1): `spatial_dim`
    fn num_components(&self) -> usize;

    /// Total number of global DOFs across the entire mesh.
    fn num_global_dofs(&self, mesh: &Mesh) -> usize;

    /// Number of local DOFs on a single element of the given type.
    fn num_local_dofs(&self, elem_type: ElementType) -> usize;

    /// Map element-local DOF indices to global DOF indices.
    fn local_to_global(&self, mesh: &Mesh, elem_idx: usize) -> Vec<usize>;

    /// Check the global DOF numbering contract for every mesh element.
    ///
    /// This is intentionally opt-in so legacy assembly remains allocation-free,
    /// while checked applications can reject malformed topology before basis
    /// tabulation or boundary elimination.
    fn validate(&self, mesh: &Mesh) -> Result<(), SpaceValidationError> {
        validate_function_space(self, mesh)
    }

    /// Tabulate all basis functions at a reference-domain point on a physical element.
    ///
    /// Returns values AND derivatives (gradients or curls) transformed to physical space
    /// via the Jacobian.
    fn tabulate(
        &self,
        mesh: &Mesh,
        elem_idx: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> BasisTabulation;

    /// Checked basis tabulation.  Spaces with a pointwise Piola contract
    /// override this to reject malformed or singular geometry without
    /// panicking; the compatibility default preserves legacy behavior for
    /// spaces whose tabulator has no typed geometry error yet.
    fn try_tabulate(
        &self,
        mesh: &Mesh,
        elem_idx: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> Result<BasisTabulation, TabulationError> {
        self.validate(mesh)?;
        Ok(self.tabulate(mesh, elem_idx, xi, eta, zeta))
    }

    /// Internal checked-tabulation hook for spaces with a stronger
    /// geometry/Piola contract. H(curl) and H(div) override this method;
    /// legacy spaces retain the default compatibility behavior.
    fn tabulate_result(
        &self,
        mesh: &Mesh,
        elem_idx: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> Result<BasisTabulation, TabulationError> {
        self.try_tabulate(mesh, elem_idx, xi, eta, zeta)
    }

    /// Collect all global DOF indices on boundary faces with the given marker.
    fn boundary_dofs(&self, mesh: &Mesh, boundary_marker: i32) -> Vec<usize>;

    /// Physical coordinate of a global nodal DOF, when the space has one.
    /// Vector components return their shared scalar-node coordinate.
    fn dof_coordinate(&self, _mesh: &Mesh, _dof: usize) -> Option<[f64; 3]> {
        None
    }

    /// Coordinates for every global nodal DOF in global ordering.
    /// Implementations should override this when bulk topology construction is
    /// substantially cheaper than repeated [`Self::dof_coordinate`] calls.
    fn dof_coordinates(&self, mesh: &Mesh) -> Option<Vec<[f64; 3]>> {
        (0..self.num_global_dofs(mesh))
            .map(|dof| self.dof_coordinate(mesh, dof))
            .collect()
    }

    /// Polynomial degree of the basis (1 for P1/Q1, 2 for P2/Q2).
    ///
    /// Used to pick quadrature orders in time-stepping assembly. The default
    /// preserves the legacy order-2 quadrature behaviour.
    fn polynomial_degree(&self) -> usize {
        1
    }
}

/// Run the common function-space topology and DOF checks.
///
/// Individual spaces may add admission checks before calling this helper when
/// their basis has a narrower geometry or element-family contract.
pub(crate) fn validate_mesh_connectivity(mesh: &Mesh) -> Result<(), SpaceValidationError> {
    for (element, mesh_element) in mesh.elements.iter().enumerate() {
        let expected_vertices = mesh_element.element_type.num_vertices();
        if mesh_element.nodes.len() < expected_vertices {
            return Err(SpaceValidationError::InvalidMesh {
                element,
                reason: "element has too few node indices",
            });
        }
        if mesh_element
            .nodes
            .iter()
            .any(|&node| node >= mesh.nodes.len())
        {
            return Err(SpaceValidationError::InvalidMesh {
                element,
                reason: "element references node outside mesh",
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_function_space<S: FunctionSpace + ?Sized>(
    space: &S,
    mesh: &Mesh,
) -> Result<(), SpaceValidationError> {
    validate_mesh_connectivity(mesh)?;
    for (element, mesh_element) in mesh.elements.iter().enumerate() {
        let expected_vertices = mesh_element.element_type.num_vertices();
        if mesh_element.nodes.len() < expected_vertices {
            return Err(SpaceValidationError::InvalidMesh {
                element,
                reason: "element has too few node indices",
            });
        }
        if mesh_element
            .nodes
            .iter()
            .any(|&node| node >= mesh.nodes.len())
        {
            return Err(SpaceValidationError::InvalidMesh {
                element,
                reason: "element references a node outside the mesh",
            });
        }
    }
    let num_global_dofs = space.num_global_dofs(mesh);
    for (element, mesh_element) in mesh.elements.iter().enumerate() {
        let dofs = space.local_to_global(mesh, element);
        let expected = space.num_local_dofs(mesh_element.element_type);
        if dofs.len() != expected {
            return Err(SpaceValidationError::LocalDofCount {
                element,
                expected,
                actual: dofs.len(),
            });
        }
        if let Some(&dof) = dofs.iter().find(|&&dof| dof >= num_global_dofs) {
            return Err(SpaceValidationError::GlobalDofOutOfBounds {
                element,
                dof,
                num_global_dofs,
            });
        }
        let mut seen = std::collections::HashSet::with_capacity(dofs.len());
        for &dof in &dofs {
            if !seen.insert(dof) {
                return Err(SpaceValidationError::DuplicateLocalDof { element, dof });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{DGSpace, H1Space, HcurlSpace, HdivSpace};

    #[test]
    fn supported_spaces_validate_triangle_dof_maps() {
        let mesh = unit_square_triangles(2);
        assert!(H1Space::new(PolynomialDegree::P1).validate(&mesh).is_ok());
        assert!(DGSpace::new(PolynomialDegree::P1).validate(&mesh).is_ok());
        assert!(HcurlSpace::new().validate(&mesh).is_ok());
        assert!(HdivSpace::new().validate(&mesh).is_ok());
    }

    #[test]
    fn space_validation_rejects_malformed_connectivity_without_panicking() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes.truncate(2);

        assert_eq!(
            H1Space::new(PolynomialDegree::P1).validate(&mesh),
            Err(SpaceValidationError::InvalidMesh {
                element: 0,
                reason: "element has too few node indices",
            })
        );
    }
}
