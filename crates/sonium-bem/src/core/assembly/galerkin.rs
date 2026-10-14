//! Genuine Galerkin assembly for the weakly singular single-layer operator.
//!
//! This module is deliberately separate from collocation assembly. Both test
//! and trial panels are integrated, with dedicated Duffy/Sauter--Schwab paths
//! for self, common-edge, and common-vertex pairs. The bounded production
//! slices cover constant/linear/quadratic affine `Tri3` single-layer blocks, the
//! weakly-singular normal-derivative traces needed by double-layer `K/K'`,
//! and the closed-surface P1 weak form of the hypersingular operator, plus a
//! regular separated isoparametric `Quad4` Müller pair block. Curved-surface,
//! higher-order hypersingular, and accelerated Galerkin formulations remain
//! separate APIs; dense-source compressed action and batch-solve adapters
//! are available on the affine Müller operator.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use std::collections::HashMap;

use crate::core::assembly::common::get_element_coords;
use crate::core::assembly::transmission::{
    CompressedTransmissionSolution, CompressedTransmissionTbemSystem, TransmissionCompressionError,
    TransmissionMaterialParameter, TransmissionSide,
};
use crate::core::incident::IncidentField;
use crate::core::integration::duffy::{quad4_basis_values, quad4_geometry};
use crate::core::integration::gauss::gauss_legendre;
use crate::core::integration::{
    GalerkinIntegrationResult, GalerkinQuad4WeakResult, N_GAUORDER,
    galerkin_quad4_regular_weak_form_wavenumber_derivative_with_basis,
    galerkin_quad4_regular_weak_form_with_basis, galerkin_single_layer_regular_quad4_with_basis,
    galerkin_single_layer_regular_triangle, galerkin_single_layer_regular_triangle_with_basis,
    galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis,
    galerkin_single_layer_self_quad4_with_basis, galerkin_single_layer_self_triangle,
    galerkin_single_layer_self_triangle_with_basis,
    galerkin_single_layer_touching_quad4_wavenumber_derivative_with_basis,
    galerkin_single_layer_touching_quad4_with_basis, sauter_schwab_galerkin_common_edge,
    sauter_schwab_galerkin_common_vertex, triangle_quadrature,
};
use crate::core::mesh::basis_values;
use crate::core::postprocess::pressure::FieldEvaluationError;
use crate::core::types::{
    AdjacencyType, BasisOrder, Element, ElementType, PhysicsParams, PhysicsParamsError,
    QuadratureConvergenceError, QuadratureOrderError, detect_adjacency,
};

/// Errors returned by bounded single-layer Galerkin assembly.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinAssemblyError {
    /// No positive quadrature order was supplied.
    #[error("Galerkin quadrature order must be positive")]
    InvalidQuadratureOrder,
    /// The supplied physical parameters failed checked validation.
    #[error("invalid Galerkin physics parameters: {0}")]
    InvalidPhysics(#[from] PhysicsParamsError),
    /// At least one active panel is required.
    #[error("Galerkin assembly requires at least one active panel")]
    EmptySurface,
    /// The bounded path accepts affine triangular panels only.
    #[error(
        "Galerkin element {element} has unsupported type {element_type:?}; only Tri3 is supported"
    )]
    UnsupportedElementType {
        /// Element index in the caller mesh.
        element: usize,
        /// Element type rejected by this slice.
        element_type: ElementType,
    },
    /// The bounded path accepts one constant coefficient per active panel.
    #[error("Galerkin element {element} has {dofs} DOFs; one constant DOF is required")]
    NonConstantPanel {
        /// Element index in the caller mesh.
        element: usize,
        /// Number of supplied element DOFs.
        dofs: usize,
    },
    /// Active DOFs must be dense zero-based panel ordering.
    #[error("Galerkin element {element} DOF {actual}, expected {expected}")]
    DofLayout {
        /// Element index in the caller mesh.
        element: usize,
        /// Supplied DOF address.
        actual: usize,
        /// Required dense address.
        expected: usize,
    },
    /// The bounded implementation supports constant, linear, and quadratic
    /// single-layer Tri3 bases; other operators may admit a smaller subset.
    #[error("Galerkin basis order {basis:?} is unsupported for this Galerkin operator")]
    UnsupportedBasisOrder {
        /// Basis order requested by the caller.
        basis: BasisOrder,
    },
    /// Panel-local Tri3 panels must provide the local DOF count for the
    /// requested basis order.
    #[error("Galerkin element {element} has {dofs} local basis DOFs")]
    InvalidBasisDofs {
        /// Element index in the caller mesh.
        element: usize,
        /// Number of supplied local DOF addresses.
        dofs: usize,
    },
    /// An active element references a node outside the supplied coordinate
    /// array.
    #[error(
        "Galerkin element {element} local node {local} references node {node}, but only {node_count} nodes exist"
    )]
    NodeOutOfRange {
        /// Element index in the caller mesh.
        element: usize,
        /// Local connectivity index.
        local: usize,
        /// Referenced global node index.
        node: usize,
        /// Supplied node-array length.
        node_count: usize,
    },
    /// A Tri3 element did not provide exactly three geometric vertices.
    #[error("Galerkin element {element} has {nodes} connectivity entries; Tri3 requires three")]
    InvalidTri3Connectivity {
        /// Element index in the caller mesh.
        element: usize,
        /// Number of connectivity entries supplied.
        nodes: usize,
    },
    /// A quadrature pair produced a non-finite matrix entry.
    #[error("Galerkin entry ({row}, {column}) is non-finite")]
    NonFiniteEntry {
        /// Test row.
        row: usize,
        /// Trial column.
        column: usize,
    },
    /// The caller supplied a global matrix size that cannot contain a local DOF.
    #[error(
        "Galerkin element {element} local DOF {local} address {dof} exceeds global size {global_dof_count}"
    )]
    GlobalDofOutOfRange {
        /// Element index in caller mesh.
        element: usize,
        /// Local basis index within the element.
        local: usize,
        /// Supplied global DOF address.
        dof: usize,
        /// Requested global matrix dimension.
        global_dof_count: usize,
    },
    /// A conforming element must not map two local basis functions to one DOF.
    #[error("Galerkin element {element} repeats global DOF {dof} at local basis {local}")]
    DuplicateGlobalDof {
        /// Element index in caller mesh.
        element: usize,
        /// Repeated local basis index.
        local: usize,
        /// Repeated global DOF address.
        dof: usize,
    },
    /// Reserved for a touching-pair principal-value admission failure in
    /// callers that provide an unsupported chart/layout. The built-in
    /// affine-Tri3 path now handles common-edge and common-vertex pairs with
    /// Sauter--Schwab charts.
    #[error(
        "Galerkin double-layer principal value for touching panels {row_panel} and {column_panel} is not implemented"
    )]
    UnsupportedPrincipalValuePair {
        /// Test/source panel index.
        row_panel: usize,
        /// Trial/field panel index.
        column_panel: usize,
    },
    /// The weak hypersingular form admits continuous affine P1/P2 traces;
    /// constant traces are not conforming for this operator.
    #[error(
        "Galerkin hypersingular basis {basis:?} unsupported; continuous Linear or Quadratic Tri3 is required"
    )]
    UnsupportedHypersingularBasis {
        /// Basis order requested by the caller.
        basis: BasisOrder,
    },
    /// A local vertex did not retain one consistent shared P1 global DOF.
    #[error(
        "Galerkin hypersingular element {element} vertex {local} (node {node}) has DOF {actual}; expected {expected}"
    )]
    NonConformingHypersingularDof {
        /// Element index in the active surface.
        element: usize,
        /// Local vertex index within the Tri3.
        local: usize,
        /// Global mesh node index.
        node: usize,
        /// Address encountered in the local element map.
        actual: usize,
        /// Address previously associated with this mesh node.
        expected: usize,
    },
    /// A quadratic edge midpoint did not retain one consistent shared edge DOF.
    #[error(
        "Galerkin quadratic hypersingular element {element} edge ({first}, {second}) has DOF {actual}; expected {expected}"
    )]
    NonConformingHypersingularEdgeDof {
        /// Element index in the active surface.
        element: usize,
        /// First endpoint of the undirected edge.
        first: usize,
        /// Second endpoint of the undirected edge.
        second: usize,
        /// Address encountered in the local element map.
        actual: usize,
        /// Address previously associated with this mesh edge.
        expected: usize,
    },
    /// A quadratic vertex and edge entity reused one global DOF.
    #[error(
        "Galerkin quadratic hypersingular element {element} local basis {local} reuses global DOF {dof} across incompatible entities"
    )]
    HypersingularGlobalDofEntityConflict {
        /// Element index in the active surface.
        element: usize,
        /// Local basis index within the element.
        local: usize,
        /// Conflicting global DOF address.
        dof: usize,
    },
    /// The weak hypersingular form omits boundary-edge terms and therefore
    /// requires a closed two-manifold active surface.
    #[error(
        "Galerkin hypersingular active edge ({first}, {second}) has incidence {incidence}; a closed surface requires exactly two"
    )]
    OpenHypersingularSurface {
        /// First endpoint of the undirected edge.
        first: usize,
        /// Second endpoint of the undirected edge.
        second: usize,
        /// Number of active panels incident on the edge.
        incidence: usize,
    },
    /// Two panels share an edge with the same directed orientation.  The
    /// closed weak-form hypersingular path requires a consistently oriented
    /// two-manifold so neighboring conormals cancel correctly.
    #[error(
        "Galerkin hypersingular edge ({first}, {second}) has inconsistent winding between elements {first_element} and {second_element}"
    )]
    InconsistentHypersingularOrientation {
        /// First endpoint of the undirected edge.
        first: usize,
        /// Second endpoint of the undirected edge.
        second: usize,
        /// First incident element.
        first_element: usize,
        /// Second incident element with the same directed edge.
        second_element: usize,
    },
    /// A Tri3 element did not provide exactly three geometric vertices.
    #[error(
        "Galerkin hypersingular element {element} has {nodes} connectivity entries; Tri3 requires three"
    )]
    InvalidHypersingularConnectivity {
        /// Element index in the active surface.
        element: usize,
        /// Number of connectivity entries supplied.
        nodes: usize,
    },
    /// A Quad4 element did not provide exactly four geometric vertices.
    #[error(
        "Galerkin Quad4 element {element} has {nodes} connectivity entries; Quad4 requires four"
    )]
    InvalidQuad4Connectivity {
        /// Element index in the active surface.
        element: usize,
        /// Number of connectivity entries supplied.
        nodes: usize,
    },
    /// A hypersingular element has zero or non-finite area.
    #[error("Galerkin hypersingular element {element} is degenerate")]
    DegenerateHypersingularElement {
        /// Element index in the active surface.
        element: usize,
    },
}

/// Assemble the weakly singular single-layer operator with true Galerkin
/// testing and trial integration.
///
/// Rows are test-panel DOFs and columns are trial-panel DOFs. The bounded
/// implementation accepts only constant affine `Tri3` panels with contiguous
/// one-DOF addresses. Self pairs use a nested Duffy map; adjacent pairs use
/// the existing Sauter--Schwab Galerkin charts; separated pairs use tensor
/// Gauss integration over both triangles.
pub fn assemble_galerkin_single_layer(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gauss_order: usize,
) -> Result<DenseMatrix<Complex64>, GalerkinAssemblyError> {
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let active = active_triangular_constant_layout(elements)?;
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let mut matrix = DenseMatrix::zeros(active.len(), active.len());

    for (row, &source_index) in active.iter().enumerate() {
        for (column, &field_index) in active.iter().enumerate() {
            let source = &elements[source_index];
            let field = &elements[field_index];
            let value = match detect_adjacency(source, field) {
                AdjacencyType::SelfElement => {
                    galerkin_single_layer_self_triangle(&coordinates[row], physics, gauss_order)
                }
                AdjacencyType::CommonEdge { .. } => {
                    sauter_schwab_galerkin_common_edge(
                        &coordinates[row],
                        &coordinates[column],
                        physics,
                        1,
                        1,
                        gauss_order,
                    )
                    .basis_results[0][0]
                        .g_integral
                }
                AdjacencyType::CommonVertex { .. } => {
                    sauter_schwab_galerkin_common_vertex(
                        &coordinates[row],
                        &coordinates[column],
                        physics,
                        1,
                        1,
                        gauss_order,
                    )
                    .basis_results[0][0]
                        .g_integral
                }
                AdjacencyType::Separated => galerkin_single_layer_regular_triangle(
                    &coordinates[row],
                    &coordinates[column],
                    physics,
                    gauss_order,
                ),
            };
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
            }
            matrix[[row, column]] = value;
        }
    }
    Ok(matrix)
}

#[allow(dead_code)]
fn assemble_galerkin_single_layer_quad4_with_basis(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
) -> Result<DenseMatrix<Complex64>, GalerkinAssemblyError> {
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let local_basis_count = match basis {
        BasisOrder::Constant => 1,
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
    };
    let active = active_quadrilateral_layout(elements, local_basis_count)?;
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let dimension = active.len() * local_basis_count;
    let mut matrix = DenseMatrix::zeros(dimension, dimension);
    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let adjacency = detect_adjacency(&elements[source_index], &elements[field_index]);
            let integrals = match adjacency {
                AdjacencyType::SelfElement => galerkin_single_layer_self_quad4_with_basis(
                    &coordinates[row_panel],
                    physics,
                    local_basis_count,
                    gauss_order,
                ),
                AdjacencyType::Separated => galerkin_single_layer_regular_quad4_with_basis(
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                ),
                AdjacencyType::CommonEdge { .. } | AdjacencyType::CommonVertex { .. } => {
                    galerkin_single_layer_touching_quad4_with_basis(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        &elements[source_index].connectivity,
                        &elements[field_index].connectivity,
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
            };
            for local_row in 0..local_basis_count {
                for local_column in 0..local_basis_count {
                    let value = integrals.basis_results[local_row][local_column].g_integral;
                    let row = row_panel * local_basis_count + local_row;
                    let column = column_panel * local_basis_count + local_column;
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    matrix[[row, column]] = value;
                }
            }
        }
    }
    Ok(matrix)
}

#[allow(dead_code)]
fn active_quadrilateral_layout(
    elements: &[Element],
    local_basis_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let mut active = Vec::new();
    for (element, value) in elements.iter().enumerate() {
        if value.property.is_evaluation() {
            continue;
        }
        if value.element_type != ElementType::Quad4 {
            return Err(GalerkinAssemblyError::UnsupportedElementType {
                element,
                element_type: value.element_type,
            });
        }
        if value.connectivity.len() != 4 {
            return Err(GalerkinAssemblyError::InvalidQuad4Connectivity {
                element,
                nodes: value.connectivity.len(),
            });
        }
        if value.dof_addresses.len() != local_basis_count {
            return Err(GalerkinAssemblyError::InvalidBasisDofs {
                element,
                dofs: value.dof_addresses.len(),
            });
        }
        let expected_base = active.len() * local_basis_count;
        for (local, &actual) in value.dof_addresses.iter().enumerate() {
            let expected = expected_base + local;
            if actual != expected {
                return Err(GalerkinAssemblyError::DofLayout {
                    element,
                    actual,
                    expected,
                });
            }
        }
        active.push(element);
    }
    if active.is_empty() {
        return Err(GalerkinAssemblyError::EmptySurface);
    }
    Ok(active)
}

fn active_triangular_constant_layout(
    elements: &[Element],
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let mut active = Vec::new();
    for (element, value) in elements.iter().enumerate() {
        if value.property.is_evaluation() {
            continue;
        }
        if value.element_type != ElementType::Tri3 {
            return Err(GalerkinAssemblyError::UnsupportedElementType {
                element,
                element_type: value.element_type,
            });
        }
        if value.dof_addresses.len() != 1 {
            return Err(GalerkinAssemblyError::NonConstantPanel {
                element,
                dofs: value.dof_addresses.len(),
            });
        }
        let expected = active.len();
        let actual = value.dof_addresses[0];
        if actual != expected {
            return Err(GalerkinAssemblyError::DofLayout {
                element,
                actual,
                expected,
            });
        }
        active.push(element);
    }
    if active.is_empty() {
        return Err(GalerkinAssemblyError::EmptySurface);
    }
    Ok(active)
}

/// Assemble the discontinuous panel-local linear-Tri3 single-layer operator.
///
/// Each active panel owns three barycentric P1 coefficients. The returned
/// matrix uses dense panel-local ordering: panel `p`, local basis `b` maps to
/// global row/column `3 * p + b`. This is intentionally discontinuous; shared
/// geometric vertices do not share algebraic DOFs in this bounded slice.
pub fn assemble_galerkin_single_layer_with_basis(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
) -> Result<DenseMatrix<Complex64>, GalerkinAssemblyError> {
    if elements.iter().any(|element| {
        !element.property.is_evaluation() && element.element_type == ElementType::Quad4
    }) {
        return assemble_galerkin_single_layer_quad4_with_basis(
            elements,
            nodes,
            physics,
            basis,
            gauss_order,
        );
    }
    if basis == BasisOrder::Constant {
        return assemble_galerkin_single_layer(elements, nodes, physics, gauss_order);
    }
    let local_basis_count = match basis {
        BasisOrder::Linear => 3,
        BasisOrder::Quadratic => 6,
        BasisOrder::Constant => unreachable!("constant basis returned above"),
    };
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let active = if basis == BasisOrder::Quadratic {
        active_triangular_quadratic_layout(elements)?
    } else {
        active_triangular_linear_layout(elements)?
    };
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let dimension = active.len() * local_basis_count;
    let mut matrix = DenseMatrix::zeros(dimension, dimension);

    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let source = &elements[source_index];
            let field = &elements[field_index];
            let result = match detect_adjacency(source, field) {
                AdjacencyType::SelfElement => galerkin_single_layer_self_triangle_with_basis(
                    &coordinates[row_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                ),
                AdjacencyType::CommonEdge { .. } => sauter_schwab_galerkin_common_edge(
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                ),
                AdjacencyType::CommonVertex { .. } => sauter_schwab_galerkin_common_vertex(
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                ),
                AdjacencyType::Separated => galerkin_single_layer_regular_triangle_with_basis(
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                ),
            };

            for source_basis in 0..local_basis_count {
                for field_basis in 0..local_basis_count {
                    let value = result.basis_results[source_basis][field_basis].g_integral;
                    let row = row_panel * local_basis_count + source_basis;
                    let column = column_panel * local_basis_count + field_basis;
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    matrix[[row, column]] = value;
                }
            }
        }
    }

    Ok(matrix)
}

/// Assemble one regular Galerkin single-layer block between two separated
/// bilinear/biquadratic `Quad4` panels.
///
/// This explicit block API is useful to callers assembling a larger
/// quadrilateral operator in their own singular-chart scheduler.  It rejects
/// topologically touching pairs before quadrature; those cases need a
/// dedicated quadrilateral Sauter--Schwab implementation and must not be
/// approximated by an ordinary tensor rule.
pub fn assemble_galerkin_single_layer_quad4_pair(
    source: &Element,
    field: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
) -> Result<DenseMatrix<Complex64>, GalerkinAssemblyError> {
    if source.element_type != ElementType::Quad4 {
        return Err(GalerkinAssemblyError::UnsupportedElementType {
            element: 0,
            element_type: source.element_type,
        });
    }
    if field.element_type != ElementType::Quad4 {
        return Err(GalerkinAssemblyError::UnsupportedElementType {
            element: 1,
            element_type: field.element_type,
        });
    }
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let adjacency = detect_adjacency(source, field);
    let local_basis_count = match basis {
        BasisOrder::Constant => 1,
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
    };
    if source.dof_addresses.len() != local_basis_count {
        return Err(GalerkinAssemblyError::InvalidBasisDofs {
            element: 0,
            dofs: source.dof_addresses.len(),
        });
    }
    if field.dof_addresses.len() != local_basis_count {
        return Err(GalerkinAssemblyError::InvalidBasisDofs {
            element: 1,
            dofs: field.dof_addresses.len(),
        });
    }
    let source_coords = get_element_coords(source, nodes);
    let field_coords = get_element_coords(field, nodes);
    let integrals = match adjacency {
        AdjacencyType::SelfElement => galerkin_single_layer_self_quad4_with_basis(
            &source_coords,
            physics,
            local_basis_count,
            gauss_order,
        ),
        AdjacencyType::Separated => galerkin_single_layer_regular_quad4_with_basis(
            &source_coords,
            &field_coords,
            physics,
            local_basis_count,
            local_basis_count,
            gauss_order,
        ),
        AdjacencyType::CommonEdge { .. } | AdjacencyType::CommonVertex { .. } => {
            galerkin_single_layer_touching_quad4_with_basis(
                &source_coords,
                &field_coords,
                &source.connectivity,
                &field.connectivity,
                physics,
                local_basis_count,
                local_basis_count,
                gauss_order,
            )
        }
    };
    let mut matrix = DenseMatrix::zeros(local_basis_count, local_basis_count);
    for row in 0..local_basis_count {
        for column in 0..local_basis_count {
            let value = integrals.basis_results[row][column].g_integral;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
            }
            matrix[[row, column]] = value;
        }
    }
    Ok(matrix)
}

/// Assemble the weakly singular single-layer operator using explicit global
/// element-to-DOF addresses.
///
/// This is the conforming counterpart to
/// [`assemble_galerkin_single_layer_with_basis`]. `Element::dof_addresses`
/// supplies the local-to-global map for every active element; addresses may be
/// shared by adjacent panels, and each local block is accumulated into the
/// caller-sized `global_dof_count` matrix. Evaluation elements are ignored in
/// the same way as the panel-local API. The bounded integration support is
/// constant and affine `Tri3` linear bases only.
pub fn assemble_galerkin_single_layer_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
) -> Result<DenseMatrix<Complex64>, GalerkinAssemblyError> {
    if elements.iter().any(|element| {
        !element.property.is_evaluation() && element.element_type == ElementType::Quad4
    }) {
        return assemble_galerkin_single_layer_quad4_global(
            elements,
            nodes,
            physics,
            basis,
            gauss_order,
            global_dof_count,
        );
    }
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;

    let local_basis_count = match basis {
        BasisOrder::Constant => 1,
        BasisOrder::Linear => 3,
        BasisOrder::Quadratic => 6,
    };
    let active =
        active_triangular_global_layout(elements, nodes, local_basis_count, global_dof_count)?;
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let mut matrix = DenseMatrix::zeros(global_dof_count, global_dof_count);

    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let source = &elements[source_index];
            let field = &elements[field_index];
            let result = match (basis, detect_adjacency(source, field)) {
                (BasisOrder::Constant, AdjacencyType::SelfElement) => {
                    galerkin_single_layer_self_triangle_with_basis(
                        &coordinates[row_panel],
                        physics,
                        1,
                        1,
                        gauss_order,
                    )
                }
                (BasisOrder::Constant, AdjacencyType::CommonEdge { .. }) => {
                    sauter_schwab_galerkin_common_edge(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        1,
                        1,
                        gauss_order,
                    )
                }
                (BasisOrder::Constant, AdjacencyType::CommonVertex { .. }) => {
                    sauter_schwab_galerkin_common_vertex(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        1,
                        1,
                        gauss_order,
                    )
                }
                (BasisOrder::Constant, AdjacencyType::Separated) => {
                    galerkin_single_layer_regular_triangle_with_basis(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        1,
                        1,
                        gauss_order,
                    )
                }
                (BasisOrder::Linear, AdjacencyType::SelfElement) => {
                    galerkin_single_layer_self_triangle_with_basis(
                        &coordinates[row_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
                (BasisOrder::Linear, AdjacencyType::CommonEdge { .. }) => {
                    sauter_schwab_galerkin_common_edge(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
                (BasisOrder::Linear, AdjacencyType::CommonVertex { .. }) => {
                    sauter_schwab_galerkin_common_vertex(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
                (BasisOrder::Linear, AdjacencyType::Separated) => {
                    galerkin_single_layer_regular_triangle_with_basis(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
                (BasisOrder::Quadratic, AdjacencyType::SelfElement) => {
                    galerkin_single_layer_self_triangle_with_basis(
                        &coordinates[row_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
                (BasisOrder::Quadratic, AdjacencyType::CommonEdge { .. }) => {
                    sauter_schwab_galerkin_common_edge(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
                (BasisOrder::Quadratic, AdjacencyType::CommonVertex { .. }) => {
                    sauter_schwab_galerkin_common_vertex(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
                (BasisOrder::Quadratic, AdjacencyType::Separated) => {
                    galerkin_single_layer_regular_triangle_with_basis(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
            };

            for local_row in 0..local_basis_count {
                for local_column in 0..local_basis_count {
                    let value = result.basis_results[local_row][local_column].g_integral;
                    let row = elements[source_index].dof_addresses[local_row];
                    let column = elements[field_index].dof_addresses[local_column];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    matrix[[row, column]] += value;
                }
            }
        }
    }

    Ok(matrix)
}

/// The two weakly singular Galerkin double-layer traces returned by the
/// bounded affine-Tri3 principal-value assembly.
#[derive(Debug, Clone)]
pub struct GalerkinDoubleLayerOperators {
    /// `K`: the field/trial normal derivative `∂G/∂n_y`.
    pub double_layer: DenseMatrix<Complex64>,
    /// `K'`: the source/test normal derivative `∂G/∂n_x`.
    pub adjoint_double_layer: DenseMatrix<Complex64>,
}

/// Assemble affine-Tri3 Galerkin double-layer traces with panel-local DOFs.
///
/// This is an intentionally bounded principal-value contract for affine Tri3
/// panel-local constant, linear, and quadratic traces. Flat affine self-panel
/// blocks are exactly zero because `r` is tangent to the panel. Separated pairs
/// use tensor Gauss integration, while common-edge and common-vertex pairs use
/// the dedicated Sauter--Schwab Galerkin charts. Those charts regularize the
/// shared-edge/shared-vertex weak singularity before the normal derivatives are
/// accumulated; no touching pair is routed through an ordinary tensor rule.
pub fn assemble_galerkin_double_layer_with_basis(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
) -> Result<GalerkinDoubleLayerOperators, GalerkinAssemblyError> {
    assemble_galerkin_double_layer_impl(elements, nodes, physics, basis, gauss_order, None)
}

/// Assemble affine-Tri3 Galerkin double-layer traces with explicit shared
/// local-to-global DOF addresses. Quadratic callers must provide six local
/// addresses per panel; repeated addresses are rejected by the layout checker.
pub fn assemble_galerkin_double_layer_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
) -> Result<GalerkinDoubleLayerOperators, GalerkinAssemblyError> {
    assemble_galerkin_double_layer_impl(
        elements,
        nodes,
        physics,
        basis,
        gauss_order,
        Some(global_dof_count),
    )
}

fn assemble_galerkin_double_layer_impl(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: Option<usize>,
) -> Result<GalerkinDoubleLayerOperators, GalerkinAssemblyError> {
    if let Some(dof_count) = global_dof_count
        && elements.iter().any(|element| {
            !element.property.is_evaluation() && element.element_type == ElementType::Quad4
        })
    {
        return assemble_galerkin_double_layer_quad4_global(
            elements,
            nodes,
            physics,
            basis,
            gauss_order,
            dof_count,
        );
    }
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;

    let local_basis_count = match basis {
        BasisOrder::Constant => 1,
        BasisOrder::Linear => 3,
        BasisOrder::Quadratic => 6,
    };
    let active = match global_dof_count {
        Some(count) => active_triangular_global_layout(elements, nodes, local_basis_count, count)?,
        None if basis == BasisOrder::Constant => active_triangular_constant_layout(elements)?,
        None => active_triangular_basis_layout(elements, local_basis_count)?,
    };
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let dimension = global_dof_count.unwrap_or(active.len() * local_basis_count);
    let mut double_layer = DenseMatrix::zeros(dimension, dimension);
    let mut adjoint_double_layer = DenseMatrix::zeros(dimension, dimension);

    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let result = match detect_adjacency(&elements[source_index], &elements[field_index]) {
                AdjacencyType::SelfElement => None,
                AdjacencyType::CommonEdge { .. } => Some(sauter_schwab_galerkin_common_edge(
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                )),
                AdjacencyType::CommonVertex { .. } => Some(sauter_schwab_galerkin_common_vertex(
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                )),
                AdjacencyType::Separated => {
                    Some(galerkin_single_layer_regular_triangle_with_basis(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    ))
                }
            };

            for local_row in 0..local_basis_count {
                let row = global_dof_count
                    .map(|_| elements[source_index].dof_addresses[local_row])
                    .unwrap_or(row_panel * local_basis_count + local_row);
                for local_column in 0..local_basis_count {
                    let column = global_dof_count
                        .map(|_| elements[field_index].dof_addresses[local_column])
                        .unwrap_or(column_panel * local_basis_count + local_column);
                    let (value, adjoint_value) = result
                        .as_ref()
                        .map(|integrals| {
                            let entry = &integrals.basis_results[local_row][local_column];
                            (entry.dg_dn_integral, entry.dg_dnx_integral)
                        })
                        .unwrap_or((Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)));
                    if !value.re.is_finite()
                        || !value.im.is_finite()
                        || !adjoint_value.re.is_finite()
                        || !adjoint_value.im.is_finite()
                    {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    double_layer[[row, column]] += value;
                    adjoint_double_layer[[row, column]] += adjoint_value;
                }
            }
        }
    }

    Ok(GalerkinDoubleLayerOperators {
        double_layer,
        adjoint_double_layer,
    })
}

/// The weakly singular bilinear form used for the conforming P1 hypersingular
/// operator.
#[derive(Debug, Clone)]
pub struct GalerkinHypersingularOperators {
    /// Finite-part `W` represented through its surface-gradient form.
    pub hypersingular: DenseMatrix<Complex64>,
}

/// Errors returned while composing the two-medium Galerkin Müller operator.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinTransmissionAssemblyError {
    /// The exterior-medium Galerkin block failed admission or integration.
    #[error("exterior Galerkin transmission block failed: {0}")]
    Exterior(#[source] GalerkinAssemblyError),
    /// The interior-medium Galerkin block failed admission or integration.
    #[error("interior Galerkin transmission block failed: {0}")]
    Interior(#[source] GalerkinAssemblyError),
    /// The two media must be evaluated at one common frequency.
    #[error("Galerkin transmission frequencies differ: exterior={exterior}, interior={interior}")]
    FrequencyMismatch {
        /// Exterior-medium frequency in hertz.
        exterior: f64,
        /// Interior-medium frequency in hertz.
        interior: f64,
    },
    /// The two media must use one common time-harmonic sign convention.
    #[error(
        "Galerkin transmission harmonic factors differ: exterior={exterior}, interior={interior}"
    )]
    HarmonicMismatch {
        /// Exterior time-harmonic sign factor.
        exterior: f64,
        /// Interior time-harmonic sign factor.
        interior: f64,
    },
    /// The weak hypersingular block requires a conforming non-constant trace basis.
    #[error("Galerkin Müller transmission basis {basis:?} is unsupported; use Linear or Quadratic")]
    UnsupportedBasis {
        /// Basis order requested by the caller.
        basis: BasisOrder,
    },
    /// A curved Quad4 pair block failed checked assembly.
    #[error(
        "curved Quad4 Galerkin transmission pair ({source_element}, {field_element}) failed: {source}"
    )]
    Quad4Pair {
        /// Source/trial element index in the global active surface.
        source_element: usize,
        /// Field/test element index in the global active surface.
        field_element: usize,
        /// Pair-level checked failure.
        #[source]
        source: GalerkinQuad4TransmissionError,
    },
}

/// Adjacent-order admission policy for checked global Galerkin Müller assembly.
///
/// The unchecked assembly entry point remains available for compatibility and
/// accepts one caller-selected order.  This policy drives two complete global
/// assemblies (coarse and fine), including self, touching, separated, curved,
/// double-layer, single-layer, and weak-hypersingular blocks, then admits the
/// fine result only when every matrix entry satisfies the mixed error budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GalerkinQuadraturePolicy {
    /// Coarse Gauss--Legendre order used for the first global assembly.
    pub coarse_order: usize,
    /// Fine Gauss--Legendre order used for the admission assembly.
    pub fine_order: usize,
    /// Absolute entrywise error budget.
    pub absolute_tolerance: f64,
    /// Relative entrywise error budget.
    pub relative_tolerance: f64,
}

impl Default for GalerkinQuadraturePolicy {
    fn default() -> Self {
        Self {
            coarse_order: 3,
            fine_order: 4,
            absolute_tolerance: 1.0e-8,
            relative_tolerance: 1.0e-6,
        }
    }
}

impl GalerkinQuadraturePolicy {
    fn validate(self) -> Result<(), QuadratureConvergenceError> {
        if self.coarse_order == 0
            || self.fine_order <= self.coarse_order
            || self.fine_order > N_GAUORDER
        {
            return Err(QuadratureConvergenceError::InvalidOrder(
                QuadratureOrderError {
                    coarse_order: self.coarse_order,
                    fine_order: self.fine_order,
                },
            ));
        }
        if !self.absolute_tolerance.is_finite()
            || !self.relative_tolerance.is_finite()
            || self.absolute_tolerance < 0.0
            || self.relative_tolerance < 0.0
        {
            return Err(QuadratureConvergenceError::InvalidTolerance {
                absolute_tolerance: self.absolute_tolerance,
                relative_tolerance: self.relative_tolerance,
            });
        }
        Ok(())
    }
}

/// Errors returned by checked adjacent-order global Galerkin assembly.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinQuadratureAdmissionError {
    /// The requested order/tolerance policy is invalid.
    #[error("invalid Galerkin quadrature policy: {0}")]
    Policy(#[source] QuadratureConvergenceError),
    /// Coarse or fine global assembly failed its existing typed admission.
    #[error("Galerkin quadrature admission assembly failed: {0}")]
    Assembly(#[source] GalerkinTransmissionAssemblyError),
    /// The fine global assembly did not improve enough over the coarse one.
    #[error("Galerkin global quadrature did not converge: {0}")]
    NotConverged(#[source] QuadratureConvergenceError),
}

/// Admission and assembly errors for a curved-Quad4 Müller block.
///
/// The pair API covers separated, touching, and self pairs. The latter two
/// use the mapped weak-form singular charts provided by the integration layer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinQuad4TransmissionError {
    /// The requested trace basis is not a Quad4 linear or quadratic basis.
    #[error(
        "curved Quad4 Galerkin transmission basis {basis:?} unsupported; use Linear or Quadratic"
    )]
    UnsupportedBasis {
        /// Requested trace basis order.
        basis: BasisOrder,
    },
    /// No positive quadrature order supplied.
    #[error("curved Quad4 Galerkin quadrature order must be positive")]
    InvalidQuadratureOrder,
    /// A medium failed checked physical validation.
    #[error("invalid {medium} curved Quad4 transmission physics: {source}")]
    InvalidPhysics {
        /// Medium label used in the diagnostic.
        medium: &'static str,
        /// Underlying checked physics error.
        #[source]
        source: PhysicsParamsError,
    },
    /// Two media must share frequency.
    #[error(
        "curved Quad4 transmission frequencies differ: exterior={exterior}, interior={interior}"
    )]
    FrequencyMismatch {
        /// Exterior frequency in hertz.
        exterior: f64,
        /// Interior frequency in hertz.
        interior: f64,
    },
    /// Two media must share the harmonic sign convention.
    #[error(
        "curved Quad4 transmission harmonic factors differ: exterior={exterior}, interior={interior}"
    )]
    HarmonicMismatch {
        /// Exterior harmonic sign factor.
        exterior: f64,
        /// Interior harmonic sign factor.
        interior: f64,
    },
    /// Both pair elements must be quadrilaterals.
    #[error("curved Quad4 transmission element {element} has unsupported type {element_type:?}")]
    UnsupportedElementType {
        /// Pair element index (source `0`, field `1`).
        element: usize,
        /// Element type encountered.
        element_type: ElementType,
    },
    /// A Quad4 element must expose four corner nodes.
    #[error(
        "curved Quad4 transmission element {element} has {nodes} connectivity entries; expected 4"
    )]
    InvalidConnectivity {
        /// Pair element index (source `0`, field `1`).
        element: usize,
        /// Connectivity length encountered.
        nodes: usize,
    },
    /// A connectivity index was outside the supplied node array.
    #[error(
        "curved Quad4 transmission element {element} local node {local} references node {node}; only {node_count} nodes supplied"
    )]
    NodeOutOfRange {
        /// Pair element index (source `0`, field `1`).
        element: usize,
        /// Local connectivity slot.
        local: usize,
        /// Referenced node index.
        node: usize,
        /// Supplied node-array length.
        node_count: usize,
    },
    /// The element-local basis map does not match the requested order.
    #[error("curved Quad4 transmission element {element} has {dofs} DOFs; expected {expected}")]
    InvalidBasisDofs {
        /// Pair element index (source `0`, field `1`).
        element: usize,
        /// Number of local addresses supplied.
        dofs: usize,
        /// Required number for the selected basis.
        expected: usize,
    },
    /// Reserved compatibility error for a pair topology that a future
    /// pair-level extension may not support. Current self/touching paths use
    /// mapped singular charts and do not construct this variant.
    #[error("curved Quad4 transmission pair API does not assemble {pair} blocks")]
    UnsupportedPair {
        /// Pair topology rejected by this pair-level API (`self`).
        pair: &'static str,
    },
    /// A composed local matrix entry was not finite.
    #[error("curved Quad4 transmission entry ({row}, {column}) non-finite")]
    NonFiniteEntry {
        /// Local matrix row.
        row: usize,
        /// Local matrix column.
        column: usize,
    },
}

/// Assemble one genuine isoparametric Quad4 Galerkin Müller transmission
/// block for any source/field pair.
///
/// The bilinear (linear order) or biquadratic (quadratic order) shape
/// functions are evaluated on the physical Quad4 map.  `V`, `K`, `K'`, and
/// the weak-form `W` contribution are integrated with the same regular
/// tensor rule, then composed into the standard two-trace Müller block.  The
/// Self and touching pairs use mapped Quad4 singular charts before
/// composition; separated pairs use the regular tensor rule.
pub fn assemble_galerkin_muller_transmission_quad4_pair(
    source: &Element,
    field: &Element,
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
) -> Result<DenseMatrix<Complex64>, GalerkinQuad4TransmissionError> {
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => {
            return Err(GalerkinQuad4TransmissionError::UnsupportedBasis { basis });
        }
    };
    if gauss_order == 0 {
        return Err(GalerkinQuad4TransmissionError::InvalidQuadratureOrder);
    }
    exterior
        .validate()
        .map_err(|source| GalerkinQuad4TransmissionError::InvalidPhysics {
            medium: "exterior",
            source,
        })?;
    interior
        .validate()
        .map_err(|source| GalerkinQuad4TransmissionError::InvalidPhysics {
            medium: "interior",
            source,
        })?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinQuad4TransmissionError::FrequencyMismatch {
            exterior: exterior.frequency,
            interior: interior.frequency,
        });
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinQuad4TransmissionError::HarmonicMismatch {
            exterior: exterior.harmonic_factor,
            interior: interior.harmonic_factor,
        });
    }

    for (element_index, element) in [(0usize, source), (1usize, field)] {
        if element.element_type != ElementType::Quad4 {
            return Err(GalerkinQuad4TransmissionError::UnsupportedElementType {
                element: element_index,
                element_type: element.element_type,
            });
        }
        if element.connectivity.len() != 4 {
            return Err(GalerkinQuad4TransmissionError::InvalidConnectivity {
                element: element_index,
                nodes: element.connectivity.len(),
            });
        }
        for (local, &node) in element.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(GalerkinQuad4TransmissionError::NodeOutOfRange {
                    element: element_index,
                    local,
                    node,
                    node_count: nodes.len(),
                });
            }
        }
        if element.dof_addresses.len() != local_basis_count {
            return Err(GalerkinQuad4TransmissionError::InvalidBasisDofs {
                element: element_index,
                dofs: element.dof_addresses.len(),
                expected: local_basis_count,
            });
        }
    }
    let source_coords = get_element_coords(source, nodes);
    let field_coords = get_element_coords(field, nodes);
    let exterior_result = galerkin_quad4_weak_pair(
        source,
        field,
        &source_coords,
        &field_coords,
        exterior,
        local_basis_count,
        gauss_order,
    );
    let interior_result = galerkin_quad4_weak_pair(
        source,
        field,
        &source_coords,
        &field_coords,
        interior,
        local_basis_count,
        gauss_order,
    );
    let dimension = 2 * local_basis_count;
    let mut matrix = DenseMatrix::zeros(dimension, dimension);
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    for row in 0..local_basis_count {
        for column in 0..local_basis_count {
            let exterior_entry = &exterior_result.basis_results[row][column];
            let interior_entry = &interior_result.basis_results[row][column];
            let exterior_w = -exterior_result.curl_integrals[row][column]
                + Complex64::new(exterior.wave_number * exterior.wave_number, 0.0)
                    * exterior_result.normal_mass_integrals[row][column];
            let interior_w = -interior_result.curl_integrals[row][column]
                + Complex64::new(interior.wave_number * interior.wave_number, 0.0)
                    * interior_result.normal_mass_integrals[row][column];
            matrix[[row, column]] = interior_entry.dg_dn_integral + exterior_entry.dg_dn_integral;
            matrix[[row, local_basis_count + column]] = -(i_omega
                * (interior.density * interior_entry.g_integral
                    + exterior.density * exterior_entry.g_integral));
            matrix[[local_basis_count + row, column]] =
                interior_w / interior.density + exterior_w / exterior.density;
            matrix[[local_basis_count + row, local_basis_count + column]] =
                -i_omega * (interior_entry.dg_dnx_integral + exterior_entry.dg_dnx_integral);
        }
    }
    if let Some(index) = matrix
        .as_slice()
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(GalerkinQuad4TransmissionError::NonFiniteEntry {
            row: index / dimension,
            column: index % dimension,
        });
    }
    Ok(matrix)
}

/// Assemble an analytic wavenumber derivative of a Quad4 Müller pair.
///
/// Geometry and density are held fixed while the selected medium's
/// wavenumber differentiates the Helmholtz kernels and the weak-form
/// hypersingular `k²` term. Self and touching pairs use the same mapped
/// Duffy/Sauter--Schwab charts as the forward Quad4 assembly.
#[allow(clippy::too_many_arguments)]
pub fn assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative(
    source: &Element,
    field: &Element,
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    side: TransmissionSide,
) -> Result<DenseMatrix<Complex64>, GalerkinTransmissionMaterialDerivativeError> {
    assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative_impl(
        source,
        field,
        nodes,
        exterior,
        interior,
        basis,
        gauss_order,
        side,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative_impl(
    source: &Element,
    field: &Element,
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    side: TransmissionSide,
    weak_hypersingular: bool,
) -> Result<DenseMatrix<Complex64>, GalerkinTransmissionMaterialDerivativeError> {
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => {
            return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
                GalerkinQuad4TransmissionError::UnsupportedBasis { basis },
            ));
        }
    };
    if gauss_order == 0 {
        return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::InvalidQuadratureOrder,
        ));
    }
    exterior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::InvalidPhysics {
                medium: "exterior",
                source,
            },
        )
    })?;
    interior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::InvalidPhysics {
                medium: "interior",
                source,
            },
        )
    })?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::FrequencyMismatch {
                exterior: exterior.frequency,
                interior: interior.frequency,
            },
        ));
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::HarmonicMismatch {
                exterior: exterior.harmonic_factor,
                interior: interior.harmonic_factor,
            },
        ));
    }
    for (element_index, element) in [(0, source), (1, field)] {
        if element.element_type != ElementType::Quad4 {
            return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
                GalerkinQuad4TransmissionError::UnsupportedElementType {
                    element: element_index,
                    element_type: element.element_type,
                },
            ));
        }
        if element.connectivity.len() != 4 {
            return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
                GalerkinQuad4TransmissionError::InvalidConnectivity {
                    element: element_index,
                    nodes: element.connectivity.len(),
                },
            ));
        }
        for (local, &node) in element.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
                    GalerkinQuad4TransmissionError::NodeOutOfRange {
                        element: element_index,
                        local,
                        node,
                        node_count: nodes.len(),
                    },
                ));
            }
        }
        if element.dof_addresses.len() != local_basis_count {
            return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
                GalerkinQuad4TransmissionError::InvalidBasisDofs {
                    element: element_index,
                    dofs: element.dof_addresses.len(),
                    expected: local_basis_count,
                },
            ));
        }
    }
    let source_coords = get_element_coords(source, nodes);
    let field_coords = get_element_coords(field, nodes);
    let exterior_result = galerkin_quad4_weak_pair(
        source,
        field,
        &source_coords,
        &field_coords,
        exterior,
        local_basis_count,
        gauss_order,
    );
    let interior_result = galerkin_quad4_weak_pair(
        source,
        field,
        &source_coords,
        &field_coords,
        interior,
        local_basis_count,
        gauss_order,
    );
    let exterior_derivative = galerkin_quad4_weak_pair_wavenumber_derivative(
        source,
        field,
        &source_coords,
        &field_coords,
        exterior,
        local_basis_count,
        gauss_order,
    );
    let interior_derivative = galerkin_quad4_weak_pair_wavenumber_derivative(
        source,
        field,
        &source_coords,
        &field_coords,
        interior,
        local_basis_count,
        gauss_order,
    );

    let dimension = 2 * local_basis_count;
    let mut matrix = DenseMatrix::zeros(dimension, dimension);
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    for row in 0..local_basis_count {
        for column in 0..local_basis_count {
            let exterior_derivative_entry = &exterior_derivative.basis_results[row][column];
            let interior_derivative_entry = &interior_derivative.basis_results[row][column];
            let exterior_weak_wavenumber_derivative = -exterior_derivative.curl_integrals[row]
                [column]
                + Complex64::new(2.0 * exterior.wave_number, 0.0)
                    * exterior_result.normal_mass_integrals[row][column]
                + Complex64::new(exterior.wave_number * exterior.wave_number, 0.0)
                    * exterior_derivative.normal_mass_integrals[row][column];
            let interior_weak_wavenumber_derivative = -interior_derivative.curl_integrals[row]
                [column]
                + Complex64::new(2.0 * interior.wave_number, 0.0)
                    * interior_result.normal_mass_integrals[row][column]
                + Complex64::new(interior.wave_number * interior.wave_number, 0.0)
                    * interior_derivative.normal_mass_integrals[row][column];
            let (selected_derivative_entry, selected_weak_wavenumber_derivative, density) =
                match side {
                    TransmissionSide::Exterior => (
                        exterior_derivative_entry,
                        exterior_weak_wavenumber_derivative,
                        exterior.density,
                    ),
                    TransmissionSide::Interior => (
                        interior_derivative_entry,
                        interior_weak_wavenumber_derivative,
                        interior.density,
                    ),
                };
            let selected_wavenumber_derivative = if weak_hypersingular {
                selected_weak_wavenumber_derivative
            } else {
                selected_derivative_entry.d2g_dnxdny_integral
            };
            matrix[[row, column]] = selected_derivative_entry.dg_dn_integral;
            matrix[[row, local_basis_count + column]] =
                -(i_omega * density * selected_derivative_entry.g_integral);
            matrix[[local_basis_count + row, column]] = selected_wavenumber_derivative / density;
            matrix[[local_basis_count + row, local_basis_count + column]] =
                -i_omega * selected_derivative_entry.dg_dnx_integral;
        }
    }
    if let Some(index) = matrix
        .as_slice()
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::NonFiniteEntry {
                row: index / dimension,
                column: index % dimension,
            },
        ));
    }
    Ok(matrix)
}

/// Assemble the analytic wavenumber derivative of a global Quad4 Müller
/// transmission matrix.
///
/// The global pressure/velocity layout is identical to
/// [`assemble_galerkin_muller_transmission_with_global_dofs`].  Every local
/// pair uses the same regular, self, or touching chart as the forward
/// operator, and shared global DOFs are accumulated deterministically.
#[allow(clippy::too_many_arguments)]
pub fn assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    side: TransmissionSide,
) -> Result<DenseMatrix<Complex64>, GalerkinTransmissionMaterialDerivativeError> {
    let basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => {
            return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
                GalerkinQuad4TransmissionError::UnsupportedBasis { basis },
            ));
        }
    };
    if gauss_order == 0 {
        return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::InvalidQuadratureOrder,
        ));
    }
    exterior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Exterior,
            source,
        }
    })?;
    interior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Interior,
            source,
        }
    })?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::FrequencyMismatch {
                exterior: exterior.frequency,
                interior: interior.frequency,
            },
        ));
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
            GalerkinQuad4TransmissionError::HarmonicMismatch {
                exterior: exterior.harmonic_factor,
                interior: interior.harmonic_factor,
            },
        ));
    }

    let active =
        validate_open_quad4_hypersingular_layout(elements, nodes, basis_count, global_dof_count)
            .map_err(|source| {
                GalerkinTransmissionMaterialDerivativeError::Assembly(
                    GalerkinTransmissionAssemblyError::Exterior(source),
                )
            })?;
    let weak_hypersingular = quad4_hypersingular_surface_is_closed(elements, &active);
    let total_dofs = 2 * global_dof_count;
    let mut derivative = DenseMatrix::zeros(total_dofs, total_dofs);

    for &source_index in &active {
        for &field_index in &active {
            let local =
                assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative_impl(
                    &elements[source_index],
                    &elements[field_index],
                    nodes,
                    exterior,
                    interior,
                    basis,
                    gauss_order,
                    side,
                    weak_hypersingular,
                )?;
            let source_dofs = &elements[source_index].dof_addresses;
            let field_dofs = &elements[field_index].dof_addresses;
            for (source_local, &source_dof) in source_dofs.iter().enumerate() {
                for (field_local, &field_dof) in field_dofs.iter().enumerate() {
                    derivative[[source_dof, field_dof]] += local[[source_local, field_local]];
                    derivative[[source_dof, global_dof_count + field_dof]] +=
                        local[[source_local, basis_count + field_local]];
                    derivative[[global_dof_count + source_dof, field_dof]] +=
                        local[[basis_count + source_local, field_local]];
                    derivative[[global_dof_count + source_dof, global_dof_count + field_dof]] +=
                        local[[basis_count + source_local, basis_count + field_local]];
                }
            }
        }
    }

    // The production closed-surface Quad4 weak hypersingular assembly applies
    // the same deterministic transpose average after global accumulation.
    // Apply its derivative here as well so this path has identical symmetry
    // and fixed-topology behavior.
    for row in 0..global_dof_count {
        for column in (row + 1)..global_dof_count {
            let value: Complex64 = (derivative[[global_dof_count + row, column]]
                + derivative[[global_dof_count + column, row]])
                * 0.5;
            derivative[[global_dof_count + row, column]] = value;
            derivative[[global_dof_count + column, row]] = value;
        }
    }

    for row in 0..total_dofs {
        for column in 0..total_dofs {
            let value = derivative[[row, column]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionMaterialDerivativeError::NonFinite { row, column });
            }
        }
    }
    Ok(derivative)
}

/// Build a kernel-native Quad4 wavenumber-derivative action through block ACA.
///
/// ACA probes assemble only the requested mapped Quad4 pair blocks and cache
/// those local analytic derivatives. The global derivative is therefore not
/// materialized before compression; shared global DOFs and the production
/// closed/open hypersingular symmetry contract are retained by the entry
/// callback.
#[allow(clippy::too_many_arguments)]
pub fn compress_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    side: TransmissionSide,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    if !matches!(basis, BasisOrder::Linear | BasisOrder::Quadratic) {
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::UnsupportedBasisOrder { basis },
        )
        .into());
    }
    exterior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    interior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Interior)?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinTransmissionAssemblyError::FrequencyMismatch {
            exterior: exterior.frequency,
            interior: interior.frequency,
        }
        .into());
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinTransmissionAssemblyError::HarmonicMismatch {
            exterior: exterior.harmonic_factor,
            interior: interior.harmonic_factor,
        }
        .into());
    }
    if gauss_order == 0 || gauss_order > N_GAUORDER {
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::InvalidQuadratureOrder,
        )
        .into());
    }

    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => unreachable!(),
    };
    let active = validate_open_quad4_hypersingular_layout(
        elements,
        nodes,
        local_basis_count,
        global_dof_count,
    )
    .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    let is_closed = quad4_hypersingular_surface_is_closed(elements, &active);
    let mut incidence = vec![Vec::<(usize, usize)>::new(); global_dof_count];
    for (panel_slot, &element_index) in active.iter().enumerate() {
        for local in 0..local_basis_count {
            incidence[elements[element_index].dof_addresses[local]].push((panel_slot, local));
        }
    }
    let mut pair_cache = HashMap::<(usize, usize), DenseMatrix<Complex64>>::new();
    let mut deferred_error = None;
    let mut entry = |row: usize, column: usize| {
        if deferred_error.is_some() {
            return Complex64::new(0.0, 0.0);
        }
        match galerkin_quad4_wavenumber_derivative_entry(
            row,
            column,
            global_dof_count,
            &incidence,
            elements,
            &active,
            nodes,
            exterior,
            interior,
            basis,
            gauss_order,
            side,
            is_closed,
            &mut pair_cache,
        ) {
            Ok(value) => value,
            Err(error) => {
                deferred_error = Some(error);
                Complex64::new(0.0, 0.0)
            }
        }
    };
    let compressed = CompressedTransmissionTbemSystem::from_entry(
        global_dof_count,
        2 * global_dof_count,
        block_size,
        tolerance,
        max_rank,
        &mut entry,
    )
    .map_err(GalerkinTransmissionCompressionError::Compression)?;
    if let Some(error) = deferred_error {
        return Err(GalerkinTransmissionCompressionError::MaterialDerivative(
            error,
        ));
    }
    Ok(compressed)
}

#[allow(clippy::too_many_arguments)]
fn galerkin_quad4_wavenumber_derivative_entry(
    row: usize,
    column: usize,
    interface_dofs: usize,
    incidence: &[Vec<(usize, usize)>],
    elements: &[Element],
    active: &[usize],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    side: TransmissionSide,
    weak_hypersingular: bool,
    pair_cache: &mut HashMap<(usize, usize), DenseMatrix<Complex64>>,
) -> Result<Complex64, GalerkinTransmissionMaterialDerivativeError> {
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => unreachable!(),
    };
    let row_block = usize::from(row >= interface_dofs);
    let column_block = usize::from(column >= interface_dofs);
    let row_dof = row % interface_dofs;
    let column_dof = column % interface_dofs;
    let local_entry = |source_incidence: &[(usize, usize)],
                       field_incidence: &[(usize, usize)],
                       row_offset: usize,
                       column_offset: usize,
                       pair_cache: &mut HashMap<(usize, usize), DenseMatrix<Complex64>>|
     -> Result<Complex64, GalerkinTransmissionMaterialDerivativeError> {
        let mut value = Complex64::new(0.0, 0.0);
        for &(source_slot, source_local) in source_incidence {
            for &(field_slot, field_local) in field_incidence {
                let key = (source_slot, field_slot);
                let local = match pair_cache.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let source = &elements[active[source_slot]];
                        let field = &elements[active[field_slot]];
                        entry.insert(
                            assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative_impl(
                                source,
                                field,
                                nodes,
                                exterior,
                                interior,
                                basis,
                                gauss_order,
                                side,
                                weak_hypersingular,
                            )?,
                        )
                    }
                };
                value += local[[row_offset + source_local, column_offset + field_local]];
            }
        }
        Ok(value)
    };

    if row_block == 1 && column_block == 0 {
        let value = local_entry(
            &incidence[row_dof],
            &incidence[column_dof],
            local_basis_count,
            0,
            pair_cache,
        )?;
        if row_dof == column_dof {
            return Ok(value);
        }
        let reverse = local_entry(
            &incidence[column_dof],
            &incidence[row_dof],
            local_basis_count,
            0,
            pair_cache,
        )?;
        return Ok((value + reverse) * 0.5);
    }

    local_entry(
        &incidence[row_dof],
        &incidence[column_dof],
        row_block * local_basis_count,
        column_block * local_basis_count,
        pair_cache,
    )
}

#[allow(clippy::too_many_arguments)]
fn galerkin_quad4_muller_density_derivative_entry(
    row: usize,
    column: usize,
    interface_dofs: usize,
    incidence: &[Vec<(usize, usize)>],
    elements: &[Element],
    active: &[usize],
    coordinates: &[DenseMatrix<f64>],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    side: TransmissionSide,
    weak_hypersingular: bool,
    pair_cache: &mut HashMap<
        (usize, usize),
        (GalerkinIntegrationResult, GalerkinIntegrationResult),
    >,
) -> Result<Complex64, GalerkinTransmissionMaterialDerivativeError> {
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => unreachable!(),
    };
    let row_block = usize::from(row >= interface_dofs);
    let column_block = usize::from(column >= interface_dofs);
    let row_dof = row % interface_dofs;
    let column_dof = column % interface_dofs;
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let selected_density = match side {
        TransmissionSide::Exterior => exterior.density,
        TransmissionSide::Interior => interior.density,
    };

    let mut local_entry = |source_incidence: &[(usize, usize)],
                           field_incidence: &[(usize, usize)]|
     -> Result<Complex64, GalerkinTransmissionMaterialDerivativeError> {
        let mut value = Complex64::new(0.0, 0.0);
        for &(source_slot, source_local) in source_incidence {
            for &(field_slot, field_local) in field_incidence {
                let key = (source_slot, field_slot);
                let (exterior_pair, interior_pair) = pair_cache.entry(key).or_insert_with(|| {
                    let source = &elements[active[source_slot]];
                    let field = &elements[active[field_slot]];
                    (
                        galerkin_quad4_weak_pair(
                            source,
                            field,
                            &coordinates[source_slot],
                            &coordinates[field_slot],
                            exterior,
                            local_basis_count,
                            gauss_order,
                        ),
                        galerkin_quad4_weak_pair(
                            source,
                            field,
                            &coordinates[source_slot],
                            &coordinates[field_slot],
                            interior,
                            local_basis_count,
                            gauss_order,
                        ),
                    )
                });
                let selected_pair = match side {
                    TransmissionSide::Exterior => exterior_pair,
                    TransmissionSide::Interior => interior_pair,
                };
                let selected_entry = &selected_pair.basis_results[source_local][field_local];
                let selected_wave_number = match side {
                    TransmissionSide::Exterior => exterior.wave_number,
                    TransmissionSide::Interior => interior.wave_number,
                };
                let selected_w = if weak_hypersingular {
                    -selected_pair.curl_integrals[source_local][field_local]
                        + Complex64::new(selected_wave_number * selected_wave_number, 0.0)
                            * selected_pair.normal_mass_integrals[source_local][field_local]
                } else {
                    selected_entry.d2g_dnxdny_integral
                };
                value += match (row_block, column_block) {
                    (0, 1) => -i_omega * selected_entry.g_integral,
                    (1, 0) => -selected_w / (selected_density * selected_density),
                    _ => Complex64::new(0.0, 0.0),
                };
            }
        }
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(GalerkinTransmissionMaterialDerivativeError::NonFinite { row, column });
        }
        Ok(value)
    };

    let value = if row_block == 1 && column_block == 0 && row_dof != column_dof {
        let direct = local_entry(&incidence[row_dof], &incidence[column_dof])?;
        let reverse = local_entry(&incidence[column_dof], &incidence[row_dof])?;
        (direct + reverse) * 0.5
    } else {
        local_entry(&incidence[row_dof], &incidence[column_dof])?
    };
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(GalerkinTransmissionMaterialDerivativeError::NonFinite { row, column });
    }
    Ok(value)
}

/// Assemble a centered material derivative for one mapped Quad4 Müller pair.
///
/// The selected exterior or interior parameter is perturbed while geometry,
/// topology, frequency, and quadrature order remain fixed.
pub fn assemble_galerkin_muller_transmission_quad4_pair_material_derivative(
    source: &Element,
    field: &Element,
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
) -> Result<DenseMatrix<Complex64>, GalerkinTransmissionMaterialDerivativeError> {
    exterior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Exterior,
            source,
        }
    })?;
    interior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Interior,
            source,
        }
    })?;
    let selected = match side {
        TransmissionSide::Exterior => exterior,
        TransmissionSide::Interior => interior,
    };
    let (plus_value, minus_value, step) = match parameter {
        TransmissionMaterialParameter::Wavenumber => {
            let step = (selected.wave_number.abs() * 1.0e-5).max(1.0e-7);
            let step = step.min(selected.wave_number * 0.1);
            (
                selected.wave_number + step,
                selected.wave_number - step,
                step,
            )
        }
        TransmissionMaterialParameter::SpeedOfSound => {
            let step = (selected.speed_of_sound.abs() * 1.0e-5).max(1.0e-5);
            (
                selected.speed_of_sound + step,
                selected.speed_of_sound - step,
                step,
            )
        }
        TransmissionMaterialParameter::Density => {
            let step = (selected.density.abs() * 1.0e-5).max(1.0e-5);
            (selected.density + step, selected.density - step, step)
        }
    };
    let perturb = |medium: &PhysicsParams, value: f64| {
        let mut perturbed = medium.clone();
        match parameter {
            TransmissionMaterialParameter::Wavenumber => {
                perturbed.speed_of_sound = perturbed.omega / value;
            }
            TransmissionMaterialParameter::SpeedOfSound => {
                perturbed.speed_of_sound = value;
            }
            TransmissionMaterialParameter::Density => {
                perturbed.density = value;
            }
        }
        perturbed.wave_number = perturbed.omega / perturbed.speed_of_sound;
        perturbed.wave_length = perturbed.speed_of_sound / perturbed.frequency;
        perturbed.pressure_factor = perturbed.density * perturbed.omega * perturbed.harmonic_factor;
        perturbed
    };
    let (exterior_plus, interior_plus) = match side {
        TransmissionSide::Exterior => (perturb(exterior, plus_value), interior.clone()),
        TransmissionSide::Interior => (exterior.clone(), perturb(interior, plus_value)),
    };
    let (exterior_minus, interior_minus) = match side {
        TransmissionSide::Exterior => (perturb(exterior, minus_value), interior.clone()),
        TransmissionSide::Interior => (exterior.clone(), perturb(interior, minus_value)),
    };
    let plus = assemble_galerkin_muller_transmission_quad4_pair(
        source,
        field,
        nodes,
        &exterior_plus,
        &interior_plus,
        basis,
        gauss_order,
    )?;
    let minus = assemble_galerkin_muller_transmission_quad4_pair(
        source,
        field,
        nodes,
        &exterior_minus,
        &interior_minus,
        basis,
        gauss_order,
    )?;
    if plus.shape() != minus.shape() {
        return Err(GalerkinTransmissionMaterialDerivativeError::Dimension {
            expected: plus.nrows(),
            plus_rows: plus.nrows(),
            plus_columns: plus.ncols(),
            minus_rows: minus.nrows(),
            minus_columns: minus.ncols(),
        });
    }
    let mut derivative = DenseMatrix::zeros(plus.nrows(), plus.ncols());
    for row in 0..derivative.nrows() {
        for column in 0..derivative.ncols() {
            let value = (plus[[row, column]] - minus[[row, column]]) / (2.0 * step);
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionMaterialDerivativeError::NonFinite { row, column });
            }
            derivative[[row, column]] = value;
        }
    }
    Ok(derivative)
}

/// Failure while evaluating a Galerkin transmission material derivative.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinTransmissionMaterialDerivativeError {
    /// A medium was invalid before the perturbation could be assembled.
    #[error("{side:?} Galerkin transmission medium is invalid: {source}")]
    InvalidMedium {
        /// Medium side that failed validation.
        side: TransmissionSide,
        /// Underlying checked physics failure.
        #[source]
        source: PhysicsParamsError,
    },
    /// The perturbed Galerkin value assembly failed.
    #[error("Galerkin transmission material derivative assembly failed: {0}")]
    Assembly(#[from] GalerkinTransmissionAssemblyError),
    /// A mapped Quad4 pair failed during pair assembly.
    #[error("Galerkin Quad4 transmission material derivative pair failed: {0}")]
    Quad4(#[from] GalerkinQuad4TransmissionError),
    /// A perturbed operator changed the advertised global two-trace size.
    #[error(
        "Galerkin transmission material derivative dimensions differ: expected {expected}x{expected}, plus {plus_rows}x{plus_columns}, minus {minus_rows}x{minus_columns}"
    )]
    Dimension {
        /// Expected total two-trace dimension.
        expected: usize,
        /// Plus perturbation row count.
        plus_rows: usize,
        /// Plus perturbation column count.
        plus_columns: usize,
        /// Minus perturbation row count.
        minus_rows: usize,
        /// Minus perturbation column count.
        minus_columns: usize,
    },
    /// A derivative entry was non-finite.
    #[error("Galerkin transmission material derivative entry ({row}, {column}) is non-finite")]
    NonFinite {
        /// Matrix row containing the invalid value.
        row: usize,
        /// Matrix column containing the invalid value.
        column: usize,
    },
}

/// Failure while building an on-demand compressed Galerkin Müller operator.
///
/// The entry callback uses the same checked admission/layout errors as dense
/// Galerkin assembly, then delegates block construction and ACA validation to
/// [`CompressedTransmissionTbemSystem`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinTransmissionCompressionError {
    /// Adjacent-order global quadrature admission failed before compression.
    #[error("Galerkin transmission compression quadrature admission failed: {0}")]
    Quadrature(#[from] GalerkinQuadratureAdmissionError),
    /// Galerkin geometry, physics, or topology admission failed.
    #[error("Galerkin transmission compression assembly failed: {0}")]
    Assembly(#[from] GalerkinTransmissionAssemblyError),
    /// The compressed block-ACA constructor rejected an entry or parameter.
    #[error("Galerkin transmission compression failed: {0}")]
    Compression(#[from] TransmissionCompressionError),
    /// Material-derivative assembly failed before compression.
    #[error("Galerkin transmission material derivative failed: {0}")]
    MaterialDerivative(#[from] GalerkinTransmissionMaterialDerivativeError),
}

/// Errors returned while forming an incident-field RHS from Galerkin trace
/// coefficients.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinTransmissionRhsError {
    /// The pressure trace has the wrong number of interface coefficients.
    #[error("Galerkin transmission pressure trace has length {actual}; expected {expected}")]
    PressureLength {
        /// Required pressure coefficient count.
        expected: usize,
        /// Supplied pressure coefficient count.
        actual: usize,
    },
    /// The normal-velocity trace has the wrong number of interface coefficients.
    #[error("Galerkin transmission normal-velocity trace has length {actual}; expected {expected}")]
    NormalVelocityLength {
        /// Required normal-velocity coefficient count.
        expected: usize,
        /// Supplied normal-velocity coefficient count.
        actual: usize,
    },
    /// A supplied trace coefficient was not finite.
    #[error("Galerkin transmission {block} trace coefficient {index} is non-finite")]
    NonFiniteTrace {
        /// Trace block containing the invalid coefficient.
        block: &'static str,
        /// Coefficient index within that block.
        index: usize,
    },
    /// The formed RHS was not finite.
    #[error("Galerkin transmission RHS entry {index} is non-finite")]
    NonFiniteRhs {
        /// Flattened RHS index.
        index: usize,
    },
}

/// Errors returned while projecting an incident field onto a Galerkin trace
/// basis.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinTransmissionProjectionError {
    /// The trace basis/layout does not match the assembled operator.
    #[error("Galerkin incident trace layout is invalid: {0}")]
    Layout(#[source] GalerkinAssemblyError),
    /// The incident-field physics failed checked validation.
    #[error("invalid Galerkin incident-field physics: {0}")]
    InvalidPhysics(#[from] PhysicsParamsError),
    /// No positive, supported quadrature order was supplied.
    #[error("Galerkin incident trace quadrature order {order} is invalid; expected 1..={max}")]
    InvalidQuadratureOrder {
        /// Requested quadrature order.
        order: usize,
        /// Maximum supported quadrature order.
        max: usize,
    },
    /// An active element did not contain the three nodes required by affine
    /// Tri3 projection.
    #[error("Galerkin incident trace element {element} has {nodes} nodes; expected 3")]
    InvalidConnectivity {
        /// Element index in the caller mesh.
        element: usize,
        /// Number of connectivity nodes supplied.
        nodes: usize,
    },
    /// An element references a node outside the supplied coordinate array.
    #[error(
        "Galerkin incident trace element {element} local node {local} references node {node}, but only {node_count} nodes exist"
    )]
    NodeOutOfRange {
        /// Element index in the caller mesh.
        element: usize,
        /// Local connectivity index.
        local: usize,
        /// Referenced global node index.
        node: usize,
        /// Supplied node-array length.
        node_count: usize,
    },
    /// A node coordinate, mapped quadrature point, or normal was non-finite.
    #[error("Galerkin incident trace element {element} has non-finite geometry")]
    NonFiniteGeometry {
        /// Element index in the caller mesh.
        element: usize,
    },
    /// The affine triangle has zero (or numerically negligible) area.
    #[error("Galerkin incident trace element {element} is degenerate")]
    DegenerateElement {
        /// Element index in the caller mesh.
        element: usize,
    },
    /// A point-source singularity lies on a projection quadrature point.
    #[error(
        "Galerkin incident trace source {source_index} is singular at element {element} quadrature point {quadrature}"
    )]
    SourceSingularity {
        /// Element containing the singular quadrature point.
        element: usize,
        /// Local quadrature-point index.
        quadrature: usize,
        /// Point-source index within the incident field.
        source_index: usize,
    },
    /// A source position or amplitude was non-finite.
    #[error("Galerkin incident trace source {source_index} has non-finite data")]
    NonFiniteSource {
        /// Point-source or plane-wave index.
        source_index: usize,
    },
    /// The evaluated incident trace was non-finite.
    #[error("Galerkin incident trace {block} value {index} is non-finite")]
    NonFiniteIncident {
        /// Trace block containing the invalid value.
        block: &'static str,
        /// Flattened quadrature-value index.
        index: usize,
    },
    /// The global L2 mass matrix could not be solved.
    #[error("Galerkin incident trace mass solve failed: {0}")]
    MassSolve(String),
    /// A projected coefficient was non-finite.
    #[error("Galerkin projected {block} coefficient {index} is non-finite")]
    NonFiniteCoefficient {
        /// Trace block containing the invalid coefficient.
        block: &'static str,
        /// Global coefficient index.
        index: usize,
    },
    /// Forming the Calderón-action RHS failed after projection.
    #[error("Galerkin incident RHS formation failed: {0}")]
    Rhs(#[source] GalerkinTransmissionRhsError),
}

/// Errors returned by checked dense Galerkin Müller solve.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinTransmissionSolveError {
    /// The supplied RHS does not match the two-trace matrix dimension.
    #[error("Galerkin transmission RHS has length {actual}; expected {expected}")]
    RhsLength {
        /// Required flattened RHS length.
        expected: usize,
        /// Supplied flattened RHS length.
        actual: usize,
    },
    /// The residual tolerance is not finite and strictly positive.
    #[error("Galerkin transmission residual tolerance {value} is invalid")]
    InvalidTolerance {
        /// Invalid tolerance supplied by the caller.
        value: f64,
    },
    /// A supplied RHS coefficient was not finite.
    #[error("Galerkin transmission RHS entry {index} is non-finite")]
    NonFiniteRhs {
        /// Flattened RHS index.
        index: usize,
    },
    /// The dense factorization failed.
    #[error("Galerkin transmission dense solve failed: {0}")]
    DirectSolve(String),
    /// The factorization returned a non-finite solution.
    #[error("Galerkin transmission solution entry {index} is non-finite")]
    NonFiniteSolution {
        /// Flattened solution index.
        index: usize,
    },
    /// The returned solution did not satisfy the original matrix residual.
    #[error("Galerkin transmission residual {residual} exceeds tolerance {tolerance}")]
    Residual {
        /// Relative residual measured against the original dense matrix.
        residual: f64,
        /// Caller-provided residual budget.
        tolerance: f64,
    },
}

/// Errors returned by checked exterior field evaluation for a Galerkin
/// Müller solution.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GalerkinTransmissionFieldError {
    /// The solved two-trace vector has the wrong block layout.
    #[error("Galerkin transmission solution has length {actual}; expected {expected}")]
    SolutionLength {
        /// Required flattened solution length.
        expected: usize,
        /// Supplied flattened solution length.
        actual: usize,
    },
    /// A solved trace coefficient is not finite.
    #[error("Galerkin transmission solution entry {index} is non-finite")]
    NonFiniteSolution {
        /// Flattened solution index containing the invalid coefficient.
        index: usize,
    },
    /// Incident trace projection failed before field evaluation.
    #[error("Galerkin incident trace projection failed: {0}")]
    Projection(#[from] GalerkinTransmissionProjectionError),
    /// The common checked pressure kernel rejected the field request.
    #[error("Galerkin exterior field evaluation failed: {0}")]
    FieldEvaluation(#[source] FieldEvaluationError),
}

/// Dense two-medium Galerkin Müller transmission operator.
///
/// The unknown ordering is `[pressure traces, normal-velocity traces]`, with
/// `interface_dofs` entries in each block. The matrix is assembled from the
/// same weakly-singular `V`, principal-value `K`/`K'`, and weak-form `W`
/// operators used by the standalone Galerkin APIs. This is the bounded CPU
/// affine-Tri3 path; dense-source compressed action is available through
/// [`Self::compress`], while kernel-native, distributed, device, and
/// curved-isoparametric transmission remain separate implementations.
#[derive(Debug, Clone)]
pub struct GalerkinTransmissionOperators {
    matrix: DenseMatrix<Complex64>,
    incident_operator: DenseMatrix<Complex64>,
    interface_dofs: usize,
    basis_order: BasisOrder,
}

/// Compute a true relative residual for a dense complex operator.
///
/// The denominator floor is the smallest positive normal-safe `f64` value so
/// tiny but non-zero right-hand sides cannot be made to pass by an arbitrary
/// engineering-scale floor.
fn dense_relative_residual(
    matrix: &DenseMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let (rows, columns) = matrix.shape();
    if rows != columns || solution.len() != columns || rhs.len() != rows {
        return f64::NAN;
    }
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    let residual_norm = (0..rows)
        .map(|row| {
            let action = (0..columns)
                .map(|column| matrix[[row, column]] * solution[column])
                .sum::<Complex64>();
            (action - rhs[row]).norm_sqr()
        })
        .sum::<f64>()
        .sqrt();
    residual_norm / rhs_norm.max(f64::MIN_POSITIVE)
}

impl GalerkinTransmissionOperators {
    /// Return the assembled pressure/normal-velocity block matrix.
    pub fn matrix(&self) -> &DenseMatrix<Complex64> {
        &self.matrix
    }

    /// Number of pressure or normal-velocity interface DOFs.
    pub const fn interface_dofs(&self) -> usize {
        self.interface_dofs
    }

    /// Total number of rows in the two-trace matrix.
    pub const fn total_dofs(&self) -> usize {
        self.interface_dofs * 2
    }

    /// Basis order used for each trace block.
    pub const fn basis_order(&self) -> BasisOrder {
        self.basis_order
    }

    /// Build the checked compressed two-trace operator for this Galerkin
    /// Müller matrix.
    ///
    /// The returned action/solve contract is shared with compressed TBEM. It
    /// intentionally carries no collocation geometry, so field evaluation
    /// remains on [`Self::evaluate_exterior_field_checked`].
    pub fn compress(
        &self,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedTransmissionTbemSystem, TransmissionCompressionError> {
        CompressedTransmissionTbemSystem::from_dense_matrix(
            self.interface_dofs,
            &self.matrix,
            block_size,
            tolerance,
            max_rank,
        )
    }

    /// Form the two-block incident RHS from pressure and normal-velocity trace
    /// coefficients represented in this operator's shared DOF basis.
    ///
    /// The returned vector is `2` times the exterior Calderón action, matching
    /// [`crate::core::assembly::transmission::TransmissionTbemSystem::incident_rhs`]
    /// without assuming collocation points or a particular trace projection.
    pub fn incident_rhs_from_trace_coefficients(
        &self,
        pressure_trace: &[Complex64],
        normal_velocity_trace: &[Complex64],
    ) -> Result<Vec<Complex64>, GalerkinTransmissionRhsError> {
        let expected = self.interface_dofs;
        if pressure_trace.len() != expected {
            return Err(GalerkinTransmissionRhsError::PressureLength {
                expected,
                actual: pressure_trace.len(),
            });
        }
        if normal_velocity_trace.len() != expected {
            return Err(GalerkinTransmissionRhsError::NormalVelocityLength {
                expected,
                actual: normal_velocity_trace.len(),
            });
        }
        for (index, &value) in pressure_trace.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionRhsError::NonFiniteTrace {
                    block: "pressure",
                    index,
                });
            }
        }
        for (index, &value) in normal_velocity_trace.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionRhsError::NonFiniteTrace {
                    block: "normal velocity",
                    index,
                });
            }
        }

        let mut traces = Vec::with_capacity(2 * expected);
        traces.extend_from_slice(pressure_trace);
        traces.extend_from_slice(normal_velocity_trace);
        let mut rhs = vec![Complex64::new(0.0, 0.0); 2 * expected];
        for (row, rhs_value) in rhs.iter_mut().enumerate() {
            *rhs_value = (0..2 * expected)
                .map(|column| self.incident_operator[[row, column]] * traces[column])
                .sum::<Complex64>()
                * 2.0;
            if !rhs_value.re.is_finite() || !rhs_value.im.is_finite() {
                return Err(GalerkinTransmissionRhsError::NonFiniteRhs { index: row });
            }
        }
        Ok(rhs)
    }

    /// Evaluate the exterior total pressure represented by a solved Galerkin
    /// Müller trace vector.
    ///
    /// The solve stores total pressure and normal-velocity traces. This
    /// helper projects the incident Cauchy data into the same shared basis,
    /// subtracts it to obtain scattered traces, and then routes through the
    /// common checked field kernel. That keeps the Galerkin path's field
    /// representation identical to the collocation transmission path while
    /// rejecting malformed layouts, non-finite values, source singularities,
    /// and invalid evaluation points before entering the integration loop.
    pub fn evaluate_exterior_field_checked(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        solution: &[Complex64],
        eval_points: &[[f64; 3]],
        incident: &IncidentField,
        exterior: &PhysicsParams,
        gauss_order: usize,
    ) -> Result<Vec<crate::core::postprocess::pressure::FieldPoint>, GalerkinTransmissionFieldError>
    {
        let expected = self.total_dofs();
        if solution.len() != expected {
            return Err(GalerkinTransmissionFieldError::SolutionLength {
                expected,
                actual: solution.len(),
            });
        }
        if let Some(index) = solution
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(GalerkinTransmissionFieldError::NonFiniteSolution { index });
        }

        let (incident_pressure, incident_velocity) = self.project_incident_trace_coefficients(
            elements,
            nodes,
            incident,
            exterior,
            gauss_order,
        )?;
        let interface_dofs = self.interface_dofs;
        let scattered_pressure = solution[..interface_dofs]
            .iter()
            .zip(incident_pressure.iter())
            .map(|(&total, &incoming)| total - incoming)
            .collect::<Vec<_>>();
        let scattered_velocity = solution[interface_dofs..]
            .iter()
            .zip(incident_velocity.iter())
            .map(|(&total, &incoming)| total - incoming)
            .collect::<Vec<_>>();

        crate::core::postprocess::pressure::compute_total_field_checked(
            eval_points,
            elements,
            nodes,
            &scattered_pressure,
            Some(&scattered_velocity),
            incident,
            exterior,
            self.basis_order,
        )
        .map_err(GalerkinTransmissionFieldError::FieldEvaluation)
    }

    /// Evaluate a checked field from a compressed Galerkin transmission solve.
    ///
    /// The compressed operator intentionally stores no mesh or basis context;
    /// this adapter requires the original Galerkin geometry explicitly and
    /// routes the certified coefficient vector through the same projection,
    /// layout, finite-value, and field-kernel checks as dense solutions.
    pub fn evaluate_exterior_field_from_compressed_solution_checked(
        &self,
        solution: &CompressedTransmissionSolution,
        elements: &[Element],
        nodes: &[[f64; 3]],
        eval_points: &[[f64; 3]],
        incident: &IncidentField,
        exterior: &PhysicsParams,
        gauss_order: usize,
    ) -> Result<Vec<crate::core::postprocess::pressure::FieldPoint>, GalerkinTransmissionFieldError>
    {
        self.evaluate_exterior_field_checked(
            elements,
            nodes,
            solution.values(),
            eval_points,
            incident,
            exterior,
            gauss_order,
        )
    }

    /// Project an incident pressure and normal-velocity trace onto the
    /// operator's shared affine-Tri3 basis using a global L2 mass solve.
    ///
    /// The returned coefficient vectors are suitable for
    /// [`Self::incident_rhs_from_trace_coefficients`]. Geometry is restricted
    /// to the same affine Tri3 surface admitted by this operator; evaluation
    /// elements are ignored, while shared vertex/edge DOFs are accumulated in
    /// the supplied global layout.
    pub fn project_incident_trace_coefficients(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        incident: &IncidentField,
        physics: &PhysicsParams,
        gauss_order: usize,
    ) -> Result<(Vec<Complex64>, Vec<Complex64>), GalerkinTransmissionProjectionError> {
        let local_basis_count = match self.basis_order {
            BasisOrder::Linear => 3,
            BasisOrder::Quadratic => 6,
            BasisOrder::Constant => {
                return Err(GalerkinTransmissionProjectionError::Layout(
                    GalerkinAssemblyError::UnsupportedBasisOrder {
                        basis: BasisOrder::Constant,
                    },
                ));
            }
        };
        if !(1..=N_GAUORDER).contains(&gauss_order) {
            return Err(
                GalerkinTransmissionProjectionError::InvalidQuadratureOrder {
                    order: gauss_order,
                    max: N_GAUORDER,
                },
            );
        }
        physics.validate()?;
        let has_quad4 = elements.iter().any(|element| {
            !element.property.is_evaluation() && element.element_type == ElementType::Quad4
        });
        if has_quad4 {
            if let Some((element, element_type)) = elements
                .iter()
                .enumerate()
                .filter(|(_, value)| !value.property.is_evaluation())
                .map(|(index, value)| (index, value.element_type))
                .find(|(_, element_type)| *element_type != ElementType::Quad4)
            {
                return Err(GalerkinTransmissionProjectionError::Layout(
                    GalerkinAssemblyError::UnsupportedElementType {
                        element,
                        element_type,
                    },
                ));
            }
            return self.project_incident_trace_coefficients_quad4(
                elements,
                nodes,
                incident,
                physics,
                gauss_order,
            );
        }
        let active = active_triangular_global_layout(
            elements,
            nodes,
            local_basis_count,
            self.interface_dofs,
        )
        .map_err(GalerkinTransmissionProjectionError::Layout)?;

        let mut mass = DenseMatrix::zeros(self.interface_dofs, self.interface_dofs);
        let mut pressure_load = vec![Complex64::new(0.0, 0.0); self.interface_dofs];
        let mut velocity_load = vec![Complex64::new(0.0, 0.0); self.interface_dofs];
        let mut points = Vec::<[f64; 3]>::new();
        let mut normals = Vec::<[f64; 3]>::new();
        let mut samples = Vec::<(usize, usize, f64, Vec<f64>)>::new();

        for &element_index in &active {
            let element = &elements[element_index];
            if element.connectivity.len() != 3 {
                return Err(GalerkinTransmissionProjectionError::InvalidConnectivity {
                    element: element_index,
                    nodes: element.connectivity.len(),
                });
            }
            let mut vertices = [[0.0; 3]; 3];
            for (local, &node) in element.connectivity.iter().enumerate() {
                if node >= nodes.len() {
                    return Err(GalerkinTransmissionProjectionError::NodeOutOfRange {
                        element: element_index,
                        local,
                        node,
                        node_count: nodes.len(),
                    });
                }
                vertices[local] = nodes[node];
                if vertices[local].iter().any(|value| !value.is_finite()) {
                    return Err(GalerkinTransmissionProjectionError::NonFiniteGeometry {
                        element: element_index,
                    });
                }
            }
            let edge_01 = [
                vertices[1][0] - vertices[0][0],
                vertices[1][1] - vertices[0][1],
                vertices[1][2] - vertices[0][2],
            ];
            let edge_02 = [
                vertices[2][0] - vertices[0][0],
                vertices[2][1] - vertices[0][1],
                vertices[2][2] - vertices[0][2],
            ];
            let cross = [
                edge_01[1] * edge_02[2] - edge_01[2] * edge_02[1],
                edge_01[2] * edge_02[0] - edge_01[0] * edge_02[2],
                edge_01[0] * edge_02[1] - edge_01[1] * edge_02[0],
            ];
            let jacobian = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            if !jacobian.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteGeometry {
                    element: element_index,
                });
            }
            if jacobian <= 1.0e-15 {
                return Err(GalerkinTransmissionProjectionError::DegenerateElement {
                    element: element_index,
                });
            }
            let normal = [
                cross[0] / jacobian,
                cross[1] / jacobian,
                cross[2] / jacobian,
            ];
            for (quadrature_index, (s, t, reference_weight)) in
                triangle_quadrature(gauss_order).into_iter().enumerate()
            {
                let point = [
                    vertices[0][0] + s * edge_01[0] + t * edge_02[0],
                    vertices[0][1] + s * edge_01[1] + t * edge_02[1],
                    vertices[0][2] + s * edge_01[2] + t * edge_02[2],
                ];
                let weight = reference_weight * jacobian;
                let values = basis_values(ElementType::Tri3, self.basis_order, s, t);
                if point.iter().any(|value| !value.is_finite())
                    || normal.iter().any(|value| !value.is_finite())
                    || !weight.is_finite()
                    || values.len() != local_basis_count
                    || values.iter().any(|value| !value.is_finite())
                {
                    return Err(GalerkinTransmissionProjectionError::NonFiniteGeometry {
                        element: element_index,
                    });
                }
                points.push(point);
                normals.push(normal);
                samples.push((element_index, quadrature_index, weight, values));
            }
        }

        validate_incident_sources(incident, &points, &samples)?;
        let pressure = incident.evaluate_pressure(&points, physics);
        let normal_derivative = incident.evaluate_normal_derivative(&points, &normals, physics);
        let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
        let normal_velocity = normal_derivative
            .into_iter()
            .map(|value| value / i_omega_rho)
            .collect::<Vec<_>>();
        for (index, value) in pressure.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteIncident {
                    block: "pressure",
                    index,
                });
            }
        }
        for (index, value) in normal_velocity.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteIncident {
                    block: "normal velocity",
                    index,
                });
            }
        }

        for (sample_index, &(element_index, _quadrature_index, weight, ref values)) in
            samples.iter().enumerate()
        {
            let element = &elements[element_index];
            for (local_row, &row) in element.dof_addresses.iter().enumerate() {
                let test_value = values[local_row];
                pressure_load[row] += pressure[sample_index] * (weight * test_value);
                velocity_load[row] += normal_velocity[sample_index] * (weight * test_value);
                for (local_column, &column) in element.dof_addresses.iter().enumerate() {
                    mass[[row, column]] +=
                        Complex64::new(weight * test_value * values[local_column], 0.0);
                }
            }
        }

        let pressure_coefficients = sonium_solvers::direct::lu_solve(&mass, &pressure_load)
            .map_err(|error| GalerkinTransmissionProjectionError::MassSolve(error.to_string()))?;
        let velocity_coefficients = sonium_solvers::direct::lu_solve(&mass, &velocity_load)
            .map_err(|error| GalerkinTransmissionProjectionError::MassSolve(error.to_string()))?;
        for (index, value) in pressure_coefficients.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteCoefficient {
                    block: "pressure",
                    index,
                });
            }
        }
        for (index, value) in velocity_coefficients.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteCoefficient {
                    block: "normal velocity",
                    index,
                });
            }
        }
        Ok((pressure_coefficients, velocity_coefficients))
    }

    /// Project incident Cauchy data onto a conforming isoparametric Quad4
    /// trace space.  The same closed-surface admission used by the weak
    /// hypersingular block is applied here so the projected traces have the
    /// identical global topology as the Müller operator.
    fn project_incident_trace_coefficients_quad4(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        incident: &IncidentField,
        physics: &PhysicsParams,
        gauss_order: usize,
    ) -> Result<(Vec<Complex64>, Vec<Complex64>), GalerkinTransmissionProjectionError> {
        let basis_count = match self.basis_order {
            BasisOrder::Linear => 4,
            BasisOrder::Quadratic => 9,
            BasisOrder::Constant => {
                return Err(GalerkinTransmissionProjectionError::Layout(
                    GalerkinAssemblyError::UnsupportedBasisOrder {
                        basis: BasisOrder::Constant,
                    },
                ));
            }
        };
        let active = validate_open_quad4_hypersingular_layout(
            elements,
            nodes,
            basis_count,
            self.interface_dofs,
        )
        .map_err(GalerkinTransmissionProjectionError::Layout)?;

        // Validate node references before using get_element_coords, whose
        // intentionally lightweight contract assumes a valid mesh.
        for &element_index in &active {
            let element = &elements[element_index];
            if element.connectivity.len() != 4 {
                return Err(GalerkinTransmissionProjectionError::InvalidConnectivity {
                    element: element_index,
                    nodes: element.connectivity.len(),
                });
            }
            for (local, &node) in element.connectivity.iter().enumerate() {
                if node >= nodes.len() {
                    return Err(GalerkinTransmissionProjectionError::NodeOutOfRange {
                        element: element_index,
                        local,
                        node,
                        node_count: nodes.len(),
                    });
                }
                if nodes[node].iter().any(|value| !value.is_finite()) {
                    return Err(GalerkinTransmissionProjectionError::NonFiniteGeometry {
                        element: element_index,
                    });
                }
            }
        }

        let coordinates = active
            .iter()
            .map(|&index| get_element_coords(&elements[index], nodes))
            .collect::<Vec<_>>();
        // A nine-node Quad4 trace has nine local coefficients, so order-two
        // tensor quadrature cannot span its mass matrix (only four samples per
        // panel).  Keep the public assembly order unchanged, but use at least
        // order three for the P2 projection solve.
        let projection_gauss_order = if basis_count == 9 {
            gauss_order.max(3)
        } else {
            gauss_order
        };
        let (quadrature_points, quadrature_weights) = gauss_legendre(projection_gauss_order);
        let mut points = Vec::<[f64; 3]>::new();
        let mut normals = Vec::<[f64; 3]>::new();
        let mut samples = Vec::<(usize, usize, f64, Vec<f64>)>::new();

        for (active_index, &element_index) in active.iter().enumerate() {
            let coords = &coordinates[active_index];
            let quadrature_count = quadrature_points.len();
            for (s_index, &s) in quadrature_points.iter().enumerate() {
                for (t_index, &t) in quadrature_points.iter().enumerate() {
                    let (point, normal, jacobian) = quad4_geometry(coords, s, t);
                    let weight =
                        quadrature_weights[s_index] * quadrature_weights[t_index] * jacobian;
                    let values = quad4_basis_values(s, t, basis_count);
                    if !jacobian.is_finite()
                        || jacobian <= 1.0e-15
                        || point.iter().any(|value| !value.is_finite())
                        || normal.iter().any(|value| !value.is_finite())
                        || !weight.is_finite()
                        || values.len() != basis_count
                        || values.iter().any(|value| !value.is_finite())
                    {
                        return Err(GalerkinTransmissionProjectionError::NonFiniteGeometry {
                            element: element_index,
                        });
                    }
                    let quadrature_index = s_index * quadrature_count + t_index;
                    points.push(point);
                    normals.push(normal);
                    samples.push((element_index, quadrature_index, weight, values));
                }
            }
        }

        validate_incident_sources(incident, &points, &samples)?;
        let pressure = incident.evaluate_pressure(&points, physics);
        let normal_derivative = incident.evaluate_normal_derivative(&points, &normals, physics);
        let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
        let normal_velocity = normal_derivative
            .into_iter()
            .map(|value| value / i_omega_rho)
            .collect::<Vec<_>>();
        for (index, value) in pressure.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteIncident {
                    block: "pressure",
                    index,
                });
            }
        }
        for (index, value) in normal_velocity.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteIncident {
                    block: "normal velocity",
                    index,
                });
            }
        }

        let mut mass = DenseMatrix::zeros(self.interface_dofs, self.interface_dofs);
        let mut pressure_load = vec![Complex64::new(0.0, 0.0); self.interface_dofs];
        let mut velocity_load = vec![Complex64::new(0.0, 0.0); self.interface_dofs];
        for (sample_index, &(element_index, _, weight, ref values)) in samples.iter().enumerate() {
            let element = &elements[element_index];
            for (local_row, &row) in element.dof_addresses.iter().enumerate() {
                let test_value = values[local_row];
                pressure_load[row] += pressure[sample_index] * (weight * test_value);
                velocity_load[row] += normal_velocity[sample_index] * (weight * test_value);
                for (local_column, &column) in element.dof_addresses.iter().enumerate() {
                    mass[[row, column]] +=
                        Complex64::new(weight * test_value * values[local_column], 0.0);
                }
            }
        }

        let pressure_coefficients = sonium_solvers::direct::lu_solve(&mass, &pressure_load)
            .map_err(|error| GalerkinTransmissionProjectionError::MassSolve(error.to_string()))?;
        let velocity_coefficients = sonium_solvers::direct::lu_solve(&mass, &velocity_load)
            .map_err(|error| GalerkinTransmissionProjectionError::MassSolve(error.to_string()))?;
        for (index, value) in pressure_coefficients.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteCoefficient {
                    block: "pressure",
                    index,
                });
            }
        }
        for (index, value) in velocity_coefficients.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteCoefficient {
                    block: "normal velocity",
                    index,
                });
            }
        }
        Ok((pressure_coefficients, velocity_coefficients))
    }

    /// Project an incident field and apply the exterior Calderón action to
    /// produce the checked two-block Müller RHS in one operation.
    pub fn incident_rhs_from_field(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        incident: &IncidentField,
        physics: &PhysicsParams,
        gauss_order: usize,
    ) -> Result<Vec<Complex64>, GalerkinTransmissionProjectionError> {
        let (pressure, normal_velocity) = self.project_incident_trace_coefficients(
            elements,
            nodes,
            incident,
            physics,
            gauss_order,
        )?;
        self.incident_rhs_from_trace_coefficients(&pressure, &normal_velocity)
            .map_err(GalerkinTransmissionProjectionError::Rhs)
    }

    /// Solve the assembled dense two-trace operator and certify its original
    /// matrix residual.
    pub fn solve(
        &self,
        rhs: &[Complex64],
        residual_tolerance: f64,
    ) -> Result<Vec<Complex64>, GalerkinTransmissionSolveError> {
        let expected = self.total_dofs();
        if rhs.len() != expected {
            return Err(GalerkinTransmissionSolveError::RhsLength {
                expected,
                actual: rhs.len(),
            });
        }
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(GalerkinTransmissionSolveError::InvalidTolerance {
                value: residual_tolerance,
            });
        }
        for (index, &value) in rhs.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionSolveError::NonFiniteRhs { index });
            }
        }
        let solution = sonium_solvers::direct::lu_solve(&self.matrix, rhs)
            .map_err(|error| GalerkinTransmissionSolveError::DirectSolve(error.to_string()))?;
        for (index, &value) in solution.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionSolveError::NonFiniteSolution { index });
            }
        }
        let residual = dense_relative_residual(&self.matrix, &solution, rhs);
        if !residual.is_finite() {
            return Err(GalerkinTransmissionSolveError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        if residual > residual_tolerance {
            return Err(GalerkinTransmissionSolveError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        Ok(solution)
    }

    /// Solve the Hermitian-adjoint Galerkin Müller system with a checked
    /// residual against `Aᴴ`.
    pub fn solve_adjoint(
        &self,
        rhs: &[Complex64],
        residual_tolerance: f64,
    ) -> Result<Vec<Complex64>, GalerkinTransmissionSolveError> {
        let expected = self.total_dofs();
        if rhs.len() != expected {
            return Err(GalerkinTransmissionSolveError::RhsLength {
                expected,
                actual: rhs.len(),
            });
        }
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(GalerkinTransmissionSolveError::InvalidTolerance {
                value: residual_tolerance,
            });
        }
        if let Some(index) = rhs
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(GalerkinTransmissionSolveError::NonFiniteRhs { index });
        }

        let mut adjoint = DenseMatrix::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                adjoint[[row, column]] = self.matrix[[column, row]].conj();
            }
        }
        let solution = sonium_solvers::direct::lu_solve(&adjoint, rhs)
            .map_err(|error| GalerkinTransmissionSolveError::DirectSolve(error.to_string()))?;
        if let Some(index) = solution
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(GalerkinTransmissionSolveError::NonFiniteSolution { index });
        }

        let residual = dense_relative_residual(&adjoint, &solution, rhs);
        if !residual.is_finite() || residual > residual_tolerance {
            return Err(GalerkinTransmissionSolveError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        Ok(solution)
    }

    /// Solve a batch of forward Galerkin Müller right-hand sides.
    pub fn solve_many(
        &self,
        rhs_columns: &[Vec<Complex64>],
        residual_tolerance: f64,
    ) -> Result<Vec<Vec<Complex64>>, GalerkinTransmissionSolveError> {
        let expected = self.total_dofs();
        validate_batch_inputs(rhs_columns, expected, residual_tolerance)?;
        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }

        let rhs_count = rhs_columns.len();
        let block_len = expected.checked_mul(rhs_count).ok_or_else(|| {
            GalerkinTransmissionSolveError::DirectSolve(
                "Galerkin transmission batch dimensions overflow".to_string(),
            )
        })?;
        let mut block_rhs = vec![Complex64::new(0.0, 0.0); block_len];
        for (column, rhs) in rhs_columns.iter().enumerate() {
            for (row, &value) in rhs.iter().enumerate() {
                block_rhs[row * rhs_count + column] = value;
            }
        }

        let factor = lu_factorize(&self.matrix)
            .map_err(|error| GalerkinTransmissionSolveError::DirectSolve(error.to_string()))?;
        let block_solution = factor
            .solve_many(&block_rhs, rhs_count)
            .map_err(|error| GalerkinTransmissionSolveError::DirectSolve(error.to_string()))?;
        let mut solutions = Vec::with_capacity(rhs_count);
        for column in 0..rhs_count {
            let solution = (0..expected)
                .map(|row| block_solution[row * rhs_count + column])
                .collect::<Vec<_>>();
            validate_batch_solution(
                &self.matrix,
                &solution,
                &rhs_columns[column],
                residual_tolerance,
            )?;
            solutions.push(solution);
        }
        Ok(solutions)
    }

    /// Solve a batch of Hermitian-adjoint Galerkin Müller right-hand sides.
    pub fn solve_adjoint_many(
        &self,
        rhs_columns: &[Vec<Complex64>],
        residual_tolerance: f64,
    ) -> Result<Vec<Vec<Complex64>>, GalerkinTransmissionSolveError> {
        let expected = self.total_dofs();
        validate_batch_inputs(rhs_columns, expected, residual_tolerance)?;
        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }

        let rhs_count = rhs_columns.len();
        let block_len = expected.checked_mul(rhs_count).ok_or_else(|| {
            GalerkinTransmissionSolveError::DirectSolve(
                "Galerkin transmission batch dimensions overflow".to_string(),
            )
        })?;
        let mut block_rhs = vec![Complex64::new(0.0, 0.0); block_len];
        for (column, rhs) in rhs_columns.iter().enumerate() {
            for (row, &value) in rhs.iter().enumerate() {
                block_rhs[row * rhs_count + column] = value;
            }
        }

        let mut adjoint = DenseMatrix::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                adjoint[[row, column]] = self.matrix[[column, row]].conj();
            }
        }
        let factor = lu_factorize(&adjoint)
            .map_err(|error| GalerkinTransmissionSolveError::DirectSolve(error.to_string()))?;
        let block_solution = factor
            .solve_many(&block_rhs, rhs_count)
            .map_err(|error| GalerkinTransmissionSolveError::DirectSolve(error.to_string()))?;
        let mut solutions = Vec::with_capacity(rhs_count);
        for column in 0..rhs_count {
            let solution = (0..expected)
                .map(|row| block_solution[row * rhs_count + column])
                .collect::<Vec<_>>();
            validate_batch_solution(
                &adjoint,
                &solution,
                &rhs_columns[column],
                residual_tolerance,
            )?;
            solutions.push(solution);
        }
        Ok(solutions)
    }
}

fn validate_batch_inputs(
    rhs_columns: &[Vec<Complex64>],
    expected: usize,
    residual_tolerance: f64,
) -> Result<(), GalerkinTransmissionSolveError> {
    if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
        return Err(GalerkinTransmissionSolveError::InvalidTolerance {
            value: residual_tolerance,
        });
    }
    for rhs in rhs_columns {
        if rhs.len() != expected {
            return Err(GalerkinTransmissionSolveError::RhsLength {
                expected,
                actual: rhs.len(),
            });
        }
        if let Some(index) = rhs
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(GalerkinTransmissionSolveError::NonFiniteRhs { index });
        }
    }
    Ok(())
}

fn validate_batch_solution(
    matrix: &DenseMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
    residual_tolerance: f64,
) -> Result<(), GalerkinTransmissionSolveError> {
    if let Some(index) = solution
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(GalerkinTransmissionSolveError::NonFiniteSolution { index });
    }
    let residual = dense_relative_residual(matrix, solution, rhs);
    if !residual.is_finite() || residual > residual_tolerance {
        return Err(GalerkinTransmissionSolveError::Residual {
            residual,
            tolerance: residual_tolerance,
        });
    }
    Ok(())
}

fn validate_incident_sources(
    incident: &IncidentField,
    points: &[[f64; 3]],
    samples: &[(usize, usize, f64, Vec<f64>)],
) -> Result<(), GalerkinTransmissionProjectionError> {
    let check_plane_wave = |source: usize, direction: &[f64; 3], amplitude: &Complex64| {
        if direction.iter().any(|value| !value.is_finite())
            || !amplitude.re.is_finite()
            || !amplitude.im.is_finite()
        {
            return Err(GalerkinTransmissionProjectionError::NonFiniteSource {
                source_index: source,
            });
        }
        Ok(())
    };
    let check_point_source = |source: usize, position: &[f64; 3], strength: &Complex64| {
        if position.iter().any(|value| !value.is_finite())
            || !strength.re.is_finite()
            || !strength.im.is_finite()
        {
            return Err(GalerkinTransmissionProjectionError::NonFiniteSource {
                source_index: source,
            });
        }
        for (point_index, point) in points.iter().enumerate() {
            let dx = point[0] - position[0];
            let dy = point[1] - position[1];
            let dz = point[2] - position[2];
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if !distance_squared.is_finite() {
                return Err(GalerkinTransmissionProjectionError::NonFiniteSource {
                    source_index: source,
                });
            }
            if distance_squared <= 1.0e-20 {
                let (element, quadrature) = (samples[point_index].0, samples[point_index].1);
                return Err(GalerkinTransmissionProjectionError::SourceSingularity {
                    element,
                    quadrature,
                    source_index: source,
                });
            }
        }
        Ok(())
    };

    match incident {
        IncidentField::PlaneWave {
            direction,
            amplitude,
        } => check_plane_wave(0, direction, amplitude),
        IncidentField::PointSource { position, strength } => {
            check_point_source(0, position, strength)
        }
        IncidentField::MultiplePlaneWaves(waves) => {
            for (source, (direction, amplitude)) in waves.iter().enumerate() {
                check_plane_wave(source, direction, amplitude)?;
            }
            Ok(())
        }
        IncidentField::MultiplePointSources(sources) => {
            for (source, (position, strength)) in sources.iter().enumerate() {
                check_point_source(source, position, strength)?;
            }
            Ok(())
        }
    }
}

/// Assemble the dense two-medium Galerkin Müller transmission operator.
///
/// For media `e` (exterior) and `i` (interior), the block matrix follows the
/// production constant-panel transmission convention:
///
/// ```text
/// [ K_i + K_e, -(iωρ_i V_i + iωρ_e V_e) ]
/// [ W_i/ρ_i + W_e/ρ_e, -iω(K'_i + K'_e) ]
/// ```
///
/// `elements[*].dof_addresses` supplies the shared local-to-global map. Both
/// `Linear` (three vertex DOFs) and `Quadratic` (six vertex/edge DOFs) traces
/// are admitted; the quadratic path uses the conforming closed-surface
/// hypersingular weak form and therefore rejects open/nonconforming meshes.
/// Assemble the analytic Quad4 density derivative on a global/shared layout.
///
/// The entry helper accumulates every repeated panel incidence into the
/// global pressure/velocity blocks, preserving the same first-owner row and
/// shared-column convention as the value and wavenumber-derivative paths.
#[allow(clippy::too_many_arguments)]
fn assemble_galerkin_muller_transmission_quad4_density_derivative_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    side: TransmissionSide,
) -> Result<DenseMatrix<Complex64>, GalerkinTransmissionMaterialDerivativeError> {
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => {
            return Err(GalerkinTransmissionMaterialDerivativeError::Quad4(
                GalerkinQuad4TransmissionError::UnsupportedBasis { basis },
            ));
        }
    };
    let active = validate_open_quad4_hypersingular_layout(
        elements,
        nodes,
        local_basis_count,
        global_dof_count,
    )
    .map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::Assembly(
            GalerkinTransmissionAssemblyError::Exterior(source),
        )
    })?;
    let is_closed = quad4_hypersingular_surface_is_closed(elements, &active);
    let coordinates = active
        .iter()
        .map(|&element_index| get_element_coords(&elements[element_index], nodes))
        .collect::<Vec<_>>();
    let mut incidence = vec![Vec::<(usize, usize)>::new(); global_dof_count];
    for (panel_slot, &element_index) in active.iter().enumerate() {
        for local in 0..local_basis_count {
            let dof = elements[element_index].dof_addresses[local];
            incidence[dof].push((panel_slot, local));
        }
    }
    let mut pair_cache =
        HashMap::<(usize, usize), (GalerkinIntegrationResult, GalerkinIntegrationResult)>::new();
    let expected = 2 * global_dof_count;
    let mut derivative = DenseMatrix::zeros(expected, expected);
    for row in 0..expected {
        for column in 0..expected {
            derivative[[row, column]] = galerkin_quad4_muller_density_derivative_entry(
                row,
                column,
                global_dof_count,
                &incidence,
                elements,
                &active,
                &coordinates,
                exterior,
                interior,
                basis,
                gauss_order,
                side,
                is_closed,
                &mut pair_cache,
            )?;
        }
    }
    Ok(derivative)
}

/// Assemble a checked Galerkin transmission matrix material derivative.
///
/// The value assembler owns the complete global-DOF topology for affine Tri3
/// and mapped Quad4 Linear/Quadratic traces, including the weak-form
/// hypersingular contribution. Mapped Quad4 layouts use the analytic
/// pair-kernel derivative path for wavenumber, fixed-frequency speed, and
/// density; affine Tri3 and mixed layouts retain the topology-preserving
/// centered fallback. Every path keeps topology and quadrature order fixed.
pub fn assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
) -> Result<DenseMatrix<Complex64>, GalerkinTransmissionMaterialDerivativeError> {
    exterior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Exterior,
            source,
        }
    })?;
    interior.validate().map_err(|source| {
        GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Interior,
            source,
        }
    })?;

    // Mapped Quad4 pair kernels already differentiate all four weak-form
    // operators analytically, including the closed-surface hypersingular
    // curl/mass contribution and explicit Müller density factors. Reuse that
    // incidence-aware path for global/shared DOFs instead of materializing
    // two perturbed global matrices. Affine Tri3 and mixed layouts retain the
    // topology-preserving centered implementation below.
    let active_count = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    let quad4_only = active_count > 0
        && elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .all(|element| element.element_type == ElementType::Quad4);
    if quad4_only && matches!(basis, BasisOrder::Linear | BasisOrder::Quadratic) {
        let selected = match side {
            TransmissionSide::Exterior => exterior,
            TransmissionSide::Interior => interior,
        };
        let derivative = match parameter {
            TransmissionMaterialParameter::Wavenumber => {
                assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
                    elements,
                    nodes,
                    exterior,
                    interior,
                    basis,
                    gauss_order,
                    global_dof_count,
                    side,
                )?
            }
            TransmissionMaterialParameter::SpeedOfSound => {
                let mut derivative =
                    assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
                        elements,
                        nodes,
                        exterior,
                        interior,
                        basis,
                        gauss_order,
                        global_dof_count,
                        side,
                    )?;
                let scale = -selected.wave_number / selected.speed_of_sound;
                for value in derivative.as_mut_slice() {
                    *value *= scale;
                }
                derivative
            }
            TransmissionMaterialParameter::Density => {
                assemble_galerkin_muller_transmission_quad4_density_derivative_with_global_dofs(
                    elements,
                    nodes,
                    exterior,
                    interior,
                    basis,
                    gauss_order,
                    global_dof_count,
                    side,
                )?
            }
        };
        if let Some(index) = derivative
            .as_slice()
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(GalerkinTransmissionMaterialDerivativeError::NonFinite {
                row: index / derivative.ncols(),
                column: index % derivative.ncols(),
            });
        }
        return Ok(derivative);
    }

    let selected = match side {
        TransmissionSide::Exterior => exterior,
        TransmissionSide::Interior => interior,
    };
    let (plus_value, minus_value, step) = match parameter {
        TransmissionMaterialParameter::Wavenumber => {
            let step = (selected.wave_number.abs() * 1.0e-5).max(1.0e-7);
            let step = step.min(selected.wave_number * 0.1);
            (
                selected.wave_number + step,
                selected.wave_number - step,
                step,
            )
        }
        TransmissionMaterialParameter::SpeedOfSound => {
            let step = (selected.speed_of_sound.abs() * 1.0e-5).max(1.0e-5);
            (
                selected.speed_of_sound + step,
                selected.speed_of_sound - step,
                step,
            )
        }
        TransmissionMaterialParameter::Density => {
            let step = (selected.density.abs() * 1.0e-5).max(1.0e-5);
            (selected.density + step, selected.density - step, step)
        }
    };

    let perturb = |medium: &PhysicsParams, value: f64| {
        let mut perturbed = medium.clone();
        match parameter {
            TransmissionMaterialParameter::Wavenumber => {
                perturbed.speed_of_sound = perturbed.omega / value;
            }
            TransmissionMaterialParameter::SpeedOfSound => {
                perturbed.speed_of_sound = value;
            }
            TransmissionMaterialParameter::Density => {
                perturbed.density = value;
            }
        }
        perturbed.wave_number = perturbed.omega / perturbed.speed_of_sound;
        perturbed.wave_length = perturbed.speed_of_sound / perturbed.frequency;
        perturbed.pressure_factor = perturbed.density * perturbed.omega * perturbed.harmonic_factor;
        perturbed
    };
    let (exterior_plus, interior_plus) = match side {
        TransmissionSide::Exterior => (perturb(exterior, plus_value), interior.clone()),
        TransmissionSide::Interior => (exterior.clone(), perturb(interior, plus_value)),
    };
    let (exterior_minus, interior_minus) = match side {
        TransmissionSide::Exterior => (perturb(exterior, minus_value), interior.clone()),
        TransmissionSide::Interior => (exterior.clone(), perturb(interior, minus_value)),
    };
    let plus = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        &exterior_plus,
        &interior_plus,
        basis,
        gauss_order,
        global_dof_count,
    )?;
    let minus = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        &exterior_minus,
        &interior_minus,
        basis,
        gauss_order,
        global_dof_count,
    )?;
    let expected = 2 * global_dof_count;
    let plus_shape = plus.matrix.shape();
    let minus_shape = minus.matrix.shape();
    if plus_shape != (expected, expected) || minus_shape != (expected, expected) {
        return Err(GalerkinTransmissionMaterialDerivativeError::Dimension {
            expected,
            plus_rows: plus_shape.0,
            plus_columns: plus_shape.1,
            minus_rows: minus_shape.0,
            minus_columns: minus_shape.1,
        });
    }

    let mut derivative = DenseMatrix::zeros(expected, expected);
    for row in 0..expected {
        for column in 0..expected {
            derivative[[row, column]] =
                (plus.matrix[[row, column]] - minus.matrix[[row, column]]) / (2.0 * step);
            let value = derivative[[row, column]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(GalerkinTransmissionMaterialDerivativeError::NonFinite { row, column });
            }
        }
    }
    Ok(derivative)
}

/// Assemble the dense affine-Tri3 Galerkin Müller transmission operator.
pub fn assemble_galerkin_muller_transmission_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
) -> Result<GalerkinTransmissionOperators, GalerkinTransmissionAssemblyError> {
    if !matches!(basis, BasisOrder::Linear | BasisOrder::Quadratic) {
        return Err(GalerkinTransmissionAssemblyError::UnsupportedBasis { basis });
    }
    exterior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    interior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Interior)?;

    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinTransmissionAssemblyError::FrequencyMismatch {
            exterior: exterior.frequency,
            interior: interior.frequency,
        });
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinTransmissionAssemblyError::HarmonicMismatch {
            exterior: exterior.harmonic_factor,
            interior: interior.harmonic_factor,
        });
    }

    // The six boundary operators are independent assemblies over
    // read-only inputs; build each same-typed pair concurrently and
    // rejoin in schedule order. Each assembly is internally serial, so
    // this adds no nested parallelism. Error mapping preserves schedule
    // order: the first failing operator in schedule order reports.
    use rayon::prelude::*;
    let single_results: Vec<Result<DenseMatrix<Complex64>, GalerkinAssemblyError>> =
        [exterior, interior]
            .into_par_iter()
            .map(|physics| {
                assemble_galerkin_single_layer_with_global_dofs(
                    elements,
                    nodes,
                    physics,
                    basis,
                    gauss_order,
                    global_dof_count,
                )
            })
            .collect();
    let mut single_results = single_results.into_iter();
    let exterior_single_layer = single_results
        .next()
        .expect("two single-layer results")
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    let interior_single_layer = single_results
        .next()
        .expect("two single-layer results")
        .map_err(GalerkinTransmissionAssemblyError::Interior)?;
    let double_results: Vec<Result<GalerkinDoubleLayerOperators, GalerkinAssemblyError>> =
        [exterior, interior]
            .into_par_iter()
            .map(|physics| {
                assemble_galerkin_double_layer_with_global_dofs(
                    elements,
                    nodes,
                    physics,
                    basis,
                    gauss_order,
                    global_dof_count,
                )
            })
            .collect();
    let mut double_results = double_results.into_iter();
    let exterior_double_layer = double_results
        .next()
        .expect("two double-layer results")
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    let interior_double_layer = double_results
        .next()
        .expect("two double-layer results")
        .map_err(GalerkinTransmissionAssemblyError::Interior)?;
    let hypersingular_results: Vec<Result<GalerkinHypersingularOperators, GalerkinAssemblyError>> =
        [exterior, interior]
            .into_par_iter()
            .map(|physics| {
                assemble_galerkin_hypersingular_with_global_dofs(
                    elements,
                    nodes,
                    physics,
                    basis,
                    gauss_order,
                    global_dof_count,
                )
            })
            .collect();
    let mut hypersingular_results = hypersingular_results.into_iter();
    let exterior_hypersingular = hypersingular_results
        .next()
        .expect("two hypersingular results")
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?
        .hypersingular;
    let interior_hypersingular = hypersingular_results
        .next()
        .expect("two hypersingular results")
        .map_err(GalerkinTransmissionAssemblyError::Interior)?
        .hypersingular;

    let mut matrix = DenseMatrix::zeros(2 * global_dof_count, 2 * global_dof_count);
    let mut incident_operator = DenseMatrix::zeros(2 * global_dof_count, 2 * global_dof_count);
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let i_omega_rho_exterior = i_omega * exterior.density;
    let i_omega_rho_interior = i_omega * interior.density;
    for row in 0..global_dof_count {
        for column in 0..global_dof_count {
            matrix[[row, column]] = interior_double_layer.double_layer[[row, column]]
                + exterior_double_layer.double_layer[[row, column]];
            matrix[[row, global_dof_count + column]] = -(i_omega_rho_interior
                * interior_single_layer[[row, column]]
                + i_omega_rho_exterior * exterior_single_layer[[row, column]]);
            matrix[[global_dof_count + row, column]] = interior_hypersingular[[row, column]]
                / interior.density
                + exterior_hypersingular[[row, column]] / exterior.density;
            matrix[[global_dof_count + row, global_dof_count + column]] = -i_omega
                * (interior_double_layer.adjoint_double_layer[[row, column]]
                    + exterior_double_layer.adjoint_double_layer[[row, column]]);
            incident_operator[[row, column]] = exterior_double_layer.double_layer[[row, column]];
            incident_operator[[row, global_dof_count + column]] =
                -i_omega_rho_exterior * exterior_single_layer[[row, column]];
            incident_operator[[global_dof_count + row, column]] =
                exterior_hypersingular[[row, column]] / exterior.density;
            incident_operator[[global_dof_count + row, global_dof_count + column]] =
                -i_omega * exterior_double_layer.adjoint_double_layer[[row, column]];
        }
    }

    if matrix
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        let index = matrix
            .as_slice()
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
            .unwrap_or(0);
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::NonFiniteEntry {
                row: index / (2 * global_dof_count).max(1),
                column: index % (2 * global_dof_count).max(1),
            },
        ));
    }

    Ok(GalerkinTransmissionOperators {
        matrix,
        incident_operator,
        interface_dofs: global_dof_count,
        basis_order: basis,
    })
}

/// Assemble a global Galerkin Müller operator behind an adjacent-order gate.
///
/// Both complete global matrices are assembled before the fine result is
/// returned. This validates the assembled operator rather than only a
/// representative pair, so self, touching, separated, curved Quad4, and
/// weak-form hypersingular contributions all participate in admission.
/// A failed gate returns a typed error and never silently falls back to the
/// coarse matrix.
pub fn assemble_galerkin_muller_transmission_with_global_dofs_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    global_dof_count: usize,
    policy: GalerkinQuadraturePolicy,
) -> Result<GalerkinTransmissionOperators, GalerkinQuadratureAdmissionError> {
    policy
        .validate()
        .map_err(GalerkinQuadratureAdmissionError::Policy)?;

    let coarse = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        policy.coarse_order,
        global_dof_count,
    )
    .map_err(GalerkinQuadratureAdmissionError::Assembly)?;
    let fine = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        policy.fine_order,
        global_dof_count,
    )
    .map_err(GalerkinQuadratureAdmissionError::Assembly)?;

    let mut max_absolute = 0.0_f64;
    let mut max_relative = 0.0_f64;
    let mut converged = coarse.matrix.shape() == fine.matrix.shape();
    if converged {
        for (coarse_value, fine_value) in
            coarse.matrix.as_slice().iter().zip(fine.matrix.as_slice())
        {
            let absolute = (*fine_value - *coarse_value).norm();
            let scale = coarse_value.norm().max(fine_value.norm());
            let relative = absolute / scale.max(f64::MIN_POSITIVE);
            max_absolute = max_absolute.max(absolute);
            max_relative = max_relative.max(relative);
            converged &= absolute.is_finite()
                && relative.is_finite()
                && absolute <= policy.absolute_tolerance + policy.relative_tolerance * scale;
        }
    }
    if !converged {
        if !max_absolute.is_finite() {
            max_absolute = f64::INFINITY;
        }
        if !max_relative.is_finite() {
            max_relative = f64::INFINITY;
        }
        return Err(GalerkinQuadratureAdmissionError::NotConverged(
            QuadratureConvergenceError::NotConverged {
                max_absolute,
                max_relative,
                absolute_tolerance: policy.absolute_tolerance,
                relative_tolerance: policy.relative_tolerance,
            },
        ));
    }

    Ok(fine)
}

/// Build a compressed Galerkin Müller operator after checked global quadrature
/// admission.
///
/// The current compressed Galerkin bridge still stores the admitted dense
/// matrix as its source; this wrapper ensures compression cannot bypass the
/// same coarse/fine convergence gate used by the checked dense path.
pub fn compress_galerkin_muller_transmission_with_global_dofs_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    global_dof_count: usize,
    policy: GalerkinQuadraturePolicy,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    let operators = assemble_galerkin_muller_transmission_with_global_dofs_checked(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        global_dof_count,
        policy,
    )
    .map_err(GalerkinTransmissionCompressionError::Quadrature)?;
    operators
        .compress(block_size, tolerance, max_rank)
        .map_err(GalerkinTransmissionCompressionError::Compression)
}

/// Build a checked compressed Galerkin Müller material-derivative action.
///
/// The value path is admitted with the supplied adjacent-order quadrature
/// policy, then the existing shared-DOF material derivative is evaluated at
/// the policy's fine order and compressed with deterministic block ACA. This
/// is a bounded CPU dense-to-compressed bridge for global Linear/Quadratic
/// layouts, including Quad4 when checked adjacent-order admission passes.
/// It does not claim kernel-native compressed differentiation, device,
/// distributed, or nested-interface derivative assembly.
#[allow(clippy::too_many_arguments)]
pub fn compress_galerkin_muller_transmission_material_derivative_with_global_dofs_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    global_dof_count: usize,
    policy: GalerkinQuadraturePolicy,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    let _ = assemble_galerkin_muller_transmission_with_global_dofs_checked(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        global_dof_count,
        policy,
    )
    .map_err(GalerkinTransmissionCompressionError::Quadrature)?;

    let derivative = assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        policy.fine_order,
        global_dof_count,
        side,
        parameter,
    )
    .map_err(GalerkinTransmissionCompressionError::MaterialDerivative)?;

    CompressedTransmissionTbemSystem::from_dense_matrix(
        global_dof_count,
        &derivative,
        block_size,
        tolerance,
        max_rank,
    )
    .map_err(GalerkinTransmissionCompressionError::Compression)
}

/// Build a compressed Galerkin Müller operator from on-demand pair entries.
///
/// This path performs the same checked closed-surface admission as dense
/// Galerkin assembly, but does not allocate the `2N × 2N` Müller matrix.
/// Block entries are evaluated from the existing affine-Tri3 singular
/// quadrature kernels and passed directly to block ACA. It is intentionally a
/// bounded CPU path: curved/isoparametric, device, distributed, and field
/// evaluation adapters remain separate APIs.
/// Build a kernel-native compressed Quad4 Galerkin Müller operator after
/// adjacent-order global quadrature admission.
///
/// The admission assembles both complete dense matrices so self, touching,
/// separated, and warped isoparametric pairs are compared before the returned
/// operator switches to on-demand block-ACA entries at the fine order. This
/// keeps the checked convergence contract while avoiding a dense matrix in the
/// compressed result.
pub fn compress_galerkin_muller_transmission_quad4_with_global_dofs_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    global_dof_count: usize,
    policy: GalerkinQuadraturePolicy,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    let admitted = assemble_galerkin_muller_transmission_with_global_dofs_checked(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        global_dof_count,
        policy,
    )?;
    let compressed = compress_galerkin_muller_transmission_quad4_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        policy.fine_order,
        global_dof_count,
        block_size,
        tolerance,
        max_rank,
    )?;
    Ok(compressed.with_exact_matrix(admitted.matrix().clone()))
}

/// Build a curved-Quad4 Galerkin Müller operator directly from pair entries.
///
/// Unlike the dense-source compatibility bridge, this path retains only the
/// block-ACA representation. Each global entry is assembled from the same
/// self/touching/separated isoparametric weak-form kernels used by dense
/// Quad4 assembly, with a small pair cache so ACA probes do not repeat panel
/// integrations. The checked layout is the closed conforming layout required
/// by the weak hypersingular block.
pub fn compress_galerkin_muller_transmission_quad4_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    if !matches!(basis, BasisOrder::Linear | BasisOrder::Quadratic) {
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::UnsupportedBasisOrder { basis },
        )
        .into());
    }
    exterior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    interior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Interior)?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinTransmissionAssemblyError::FrequencyMismatch {
            exterior: exterior.frequency,
            interior: interior.frequency,
        }
        .into());
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinTransmissionAssemblyError::HarmonicMismatch {
            exterior: exterior.harmonic_factor,
            interior: interior.harmonic_factor,
        }
        .into());
    }
    if gauss_order == 0 || gauss_order > N_GAUORDER {
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::InvalidQuadratureOrder,
        )
        .into());
    }
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => unreachable!(),
    };
    let active = validate_open_quad4_hypersingular_layout(
        elements,
        nodes,
        local_basis_count,
        global_dof_count,
    )
    .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    let is_closed = quad4_hypersingular_surface_is_closed(elements, &active);
    let coordinates = active
        .iter()
        .map(|&element_index| get_element_coords(&elements[element_index], nodes))
        .collect::<Vec<_>>();
    let mut incidence = vec![Vec::<(usize, usize)>::new(); global_dof_count];
    for (panel_slot, &element_index) in active.iter().enumerate() {
        for local in 0..local_basis_count {
            incidence[elements[element_index].dof_addresses[local]].push((panel_slot, local));
        }
    }
    let mut pair_cache =
        HashMap::<(usize, usize), (GalerkinIntegrationResult, GalerkinIntegrationResult)>::new();
    let entry = |row: usize, column: usize| {
        galerkin_quad4_muller_entry(
            row,
            column,
            global_dof_count,
            &incidence,
            elements,
            &active,
            &coordinates,
            exterior,
            interior,
            basis,
            gauss_order,
            is_closed,
            &mut pair_cache,
        )
    };
    CompressedTransmissionTbemSystem::from_entry(
        global_dof_count,
        2 * global_dof_count,
        block_size,
        tolerance,
        max_rank,
        entry,
    )
    .map_err(GalerkinTransmissionCompressionError::Compression)
}

/// Build a kernel-native compressed Quad4 Galerkin Müller material-derivative action.
///
/// The entry callback evaluates analytic production Quad4 pair kernels on demand
/// while ACA requests blocks. Wavenumber and sound-speed paths share the analytic
/// kernel derivative; density differentiates the explicit Müller coefficients.
///
/// Device/distributed execution and nested-interface assembly remain separate
/// contracts.
#[allow(clippy::too_many_arguments)]
pub fn compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    compress_galerkin_muller_transmission_quad4_material_derivative_kernel_native(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        gauss_order,
        global_dof_count,
        side,
        parameter,
        block_size,
        tolerance,
        max_rank,
    )
}

#[allow(clippy::too_many_arguments)]
fn compress_galerkin_muller_transmission_quad4_material_derivative_kernel_native(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    if !matches!(basis, BasisOrder::Linear | BasisOrder::Quadratic) {
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::UnsupportedBasisOrder { basis },
        )
        .into());
    }
    exterior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    interior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Interior)?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinTransmissionAssemblyError::FrequencyMismatch {
            exterior: exterior.frequency,
            interior: interior.frequency,
        }
        .into());
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinTransmissionAssemblyError::HarmonicMismatch {
            exterior: exterior.harmonic_factor,
            interior: interior.harmonic_factor,
        }
        .into());
    }
    if gauss_order == 0 || gauss_order > N_GAUORDER {
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::InvalidQuadratureOrder,
        )
        .into());
    }
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => unreachable!(),
    };
    let active = validate_open_quad4_hypersingular_layout(
        elements,
        nodes,
        local_basis_count,
        global_dof_count,
    )
    .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    let is_closed = quad4_hypersingular_surface_is_closed(elements, &active);
    let coordinates = active
        .iter()
        .map(|&element_index| get_element_coords(&elements[element_index], nodes))
        .collect::<Vec<_>>();
    let mut incidence = vec![Vec::<(usize, usize)>::new(); global_dof_count];
    for (panel_slot, &element_index) in active.iter().enumerate() {
        for local in 0..local_basis_count {
            incidence[elements[element_index].dof_addresses[local]].push((panel_slot, local));
        }
    }
    let derivative_scale = match parameter {
        TransmissionMaterialParameter::Wavenumber => 1.0,
        TransmissionMaterialParameter::SpeedOfSound => {
            let selected = match side {
                TransmissionSide::Exterior => exterior,
                TransmissionSide::Interior => interior,
            };
            -selected.wave_number / selected.speed_of_sound
        }
        TransmissionMaterialParameter::Density => 1.0,
    };
    let mut derivative_pair_cache = HashMap::<(usize, usize), DenseMatrix<Complex64>>::new();
    let mut value_pair_cache =
        HashMap::<(usize, usize), (GalerkinIntegrationResult, GalerkinIntegrationResult)>::new();
    let mut deferred_error = None;
    let mut entry = |row: usize, column: usize| {
        if deferred_error.is_some() {
            return Complex64::new(0.0, 0.0);
        }
        let result = match parameter {
            TransmissionMaterialParameter::Density => {
                galerkin_quad4_muller_density_derivative_entry(
                    row,
                    column,
                    global_dof_count,
                    &incidence,
                    elements,
                    &active,
                    &coordinates,
                    exterior,
                    interior,
                    basis,
                    gauss_order,
                    side,
                    is_closed,
                    &mut value_pair_cache,
                )
            }
            TransmissionMaterialParameter::Wavenumber
            | TransmissionMaterialParameter::SpeedOfSound => {
                galerkin_quad4_wavenumber_derivative_entry(
                    row,
                    column,
                    global_dof_count,
                    &incidence,
                    elements,
                    &active,
                    nodes,
                    exterior,
                    interior,
                    basis,
                    gauss_order,
                    side,
                    is_closed,
                    &mut derivative_pair_cache,
                )
                .map(|value| value * derivative_scale)
            }
        };
        match result {
            Ok(value) => value,
            Err(error) => {
                deferred_error = Some(error);
                Complex64::new(0.0, 0.0)
            }
        }
    };
    let compressed = CompressedTransmissionTbemSystem::from_entry(
        global_dof_count,
        2 * global_dof_count,
        block_size,
        tolerance,
        max_rank,
        &mut entry,
    )
    .map_err(GalerkinTransmissionCompressionError::Compression)?;
    if let Some(error) = deferred_error {
        return Err(GalerkinTransmissionCompressionError::MaterialDerivative(
            error,
        ));
    }
    Ok(compressed)
}

/// Checked adjacent-order quadrature admission for the compressed Quad4
/// Galerkin Müller material-derivative action.
#[allow(clippy::too_many_arguments)]
pub fn compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    global_dof_count: usize,
    policy: GalerkinQuadraturePolicy,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    assemble_galerkin_muller_transmission_with_global_dofs_checked(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        global_dof_count,
        policy,
    )
    .map_err(GalerkinTransmissionCompressionError::Quadrature)?;
    compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        policy.fine_order,
        global_dof_count,
        side,
        parameter,
        block_size,
        tolerance,
        max_rank,
    )
}

/// Build a compressed Galerkin Müller transmission operator.
///
/// Quad4 meshes use the kernel-native on-demand path; affine triangular
/// meshes retain the dense-source compatibility bridge and its checked
/// quadrature admission wrapper.
pub fn compress_galerkin_muller_transmission_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
    block_size: usize,
    tolerance: f64,
    max_rank: usize,
) -> Result<CompressedTransmissionTbemSystem, GalerkinTransmissionCompressionError> {
    if elements.iter().any(|element| {
        !element.property.is_evaluation() && element.element_type == ElementType::Quad4
    }) {
        return compress_galerkin_muller_transmission_quad4_with_global_dofs(
            elements,
            nodes,
            exterior,
            interior,
            basis,
            gauss_order,
            global_dof_count,
            block_size,
            tolerance,
            max_rank,
        );
    }
    if !matches!(basis, BasisOrder::Linear | BasisOrder::Quadratic) {
        return Err(GalerkinTransmissionAssemblyError::UnsupportedBasis { basis }.into());
    }
    exterior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    interior
        .validate()
        .map_err(GalerkinAssemblyError::InvalidPhysics)
        .map_err(GalerkinTransmissionAssemblyError::Interior)?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(GalerkinTransmissionAssemblyError::FrequencyMismatch {
            exterior: exterior.frequency,
            interior: interior.frequency,
        }
        .into());
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(GalerkinTransmissionAssemblyError::HarmonicMismatch {
            exterior: exterior.harmonic_factor,
            interior: interior.harmonic_factor,
        }
        .into());
    }
    if gauss_order == 0 || gauss_order > N_GAUORDER {
        return Err(GalerkinTransmissionAssemblyError::Exterior(
            GalerkinAssemblyError::InvalidQuadratureOrder,
        )
        .into());
    }

    let local_basis_count = match basis {
        BasisOrder::Linear => 3,
        BasisOrder::Quadratic => 6,
        BasisOrder::Constant => unreachable!(),
    };
    let active =
        active_triangular_global_layout(elements, nodes, local_basis_count, global_dof_count)
            .map_err(GalerkinTransmissionAssemblyError::Exterior)?;
    let coordinates = active
        .iter()
        .map(|&element_index| get_element_coords(&elements[element_index], nodes))
        .collect::<Vec<_>>();
    let geometry = coordinates
        .iter()
        .enumerate()
        .map(|(slot, coords)| affine_triangle_geometry(coords, active[slot]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(GalerkinTransmissionAssemblyError::Exterior)?;

    let mut incidence = vec![Vec::<(usize, usize)>::new(); global_dof_count];
    for (panel_slot, &element_index) in active.iter().enumerate() {
        for local in 0..local_basis_count {
            let dof = elements[element_index].dof_addresses[local];
            incidence[dof].push((panel_slot, local));
        }
    }
    let is_closed = triangular_hypersingular_surface_is_closed(elements, &active);

    let entry = |row: usize, column: usize| {
        galerkin_muller_entry(
            row,
            column,
            global_dof_count,
            &incidence,
            elements,
            &active,
            &coordinates,
            &geometry,
            exterior,
            interior,
            basis,
            gauss_order,
            is_closed,
        )
    };

    CompressedTransmissionTbemSystem::from_entry(
        global_dof_count,
        2 * global_dof_count,
        block_size,
        tolerance,
        max_rank,
        entry,
    )
    .map_err(Into::into)
}

fn galerkin_quad4_muller_entry(
    row: usize,
    column: usize,
    interface_dofs: usize,
    incidence: &[Vec<(usize, usize)>],
    elements: &[Element],
    active: &[usize],
    coordinates: &[DenseMatrix<f64>],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    is_closed: bool,
    pair_cache: &mut HashMap<
        (usize, usize),
        (GalerkinIntegrationResult, GalerkinIntegrationResult),
    >,
) -> Complex64 {
    let row_block = usize::from(row >= interface_dofs);
    let column_block = usize::from(column >= interface_dofs);
    let row_dof = row % interface_dofs;
    let column_dof = column % interface_dofs;
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let mut value = Complex64::new(0.0, 0.0);
    for &(source_slot, source_local) in &incidence[row_dof] {
        for &(field_slot, field_local) in &incidence[column_dof] {
            let (exterior_pair, interior_pair) = pair_cache
                .entry((source_slot, field_slot))
                .or_insert_with(|| {
                    let source = &elements[active[source_slot]];
                    let field = &elements[active[field_slot]];
                    let source_coords = &coordinates[source_slot];
                    let field_coords = &coordinates[field_slot];
                    (
                        galerkin_quad4_weak_pair(
                            source,
                            field,
                            source_coords,
                            field_coords,
                            exterior,
                            match basis {
                                BasisOrder::Linear => 4,
                                BasisOrder::Quadratic => 9,
                                BasisOrder::Constant => unreachable!(),
                            },
                            gauss_order,
                        ),
                        galerkin_quad4_weak_pair(
                            source,
                            field,
                            source_coords,
                            field_coords,
                            interior,
                            match basis {
                                BasisOrder::Linear => 4,
                                BasisOrder::Quadratic => 9,
                                BasisOrder::Constant => unreachable!(),
                            },
                            gauss_order,
                        ),
                    )
                });
            let exterior_entry = &exterior_pair.basis_results[source_local][field_local];
            let interior_entry = &interior_pair.basis_results[source_local][field_local];
            let exterior_w = if is_closed {
                -exterior_pair.curl_integrals[source_local][field_local]
                    + Complex64::new(exterior.wave_number * exterior.wave_number, 0.0)
                        * exterior_pair.normal_mass_integrals[source_local][field_local]
            } else {
                exterior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            let interior_w = if is_closed {
                -interior_pair.curl_integrals[source_local][field_local]
                    + Complex64::new(interior.wave_number * interior.wave_number, 0.0)
                        * interior_pair.normal_mass_integrals[source_local][field_local]
            } else {
                interior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            value += match (row_block, column_block) {
                (0, 0) => interior_entry.dg_dn_integral + exterior_entry.dg_dn_integral,
                (0, 1) => {
                    -(i_omega
                        * (interior.density * interior_entry.g_integral
                            + exterior.density * exterior_entry.g_integral))
                }
                (1, 0) => interior_w / interior.density + exterior_w / exterior.density,
                (1, 1) => {
                    -i_omega * (interior_entry.dg_dnx_integral + exterior_entry.dg_dnx_integral)
                }
                _ => unreachable!(),
            };
        }
    }
    if row_block == 1 && column_block == 0 && row_dof != column_dof {
        let reverse = galerkin_quad4_hypersingular_global_entry(
            &incidence[column_dof],
            &incidence[row_dof],
            elements,
            active,
            coordinates,
            exterior,
            interior,
            basis,
            gauss_order,
            is_closed,
            pair_cache,
        );
        value = (value + reverse) * 0.5;
    }
    value
}

fn galerkin_quad4_hypersingular_global_entry(
    source_incidence: &[(usize, usize)],
    field_incidence: &[(usize, usize)],
    elements: &[Element],
    active: &[usize],
    coordinates: &[DenseMatrix<f64>],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    is_closed: bool,
    pair_cache: &mut HashMap<
        (usize, usize),
        (GalerkinIntegrationResult, GalerkinIntegrationResult),
    >,
) -> Complex64 {
    let local_basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => unreachable!(),
    };
    let mut value = Complex64::new(0.0, 0.0);
    for &(source_slot, source_local) in source_incidence {
        for &(field_slot, field_local) in field_incidence {
            let (exterior_pair, interior_pair) = pair_cache
                .entry((source_slot, field_slot))
                .or_insert_with(|| {
                    let source = &elements[active[source_slot]];
                    let field = &elements[active[field_slot]];
                    let source_coords = &coordinates[source_slot];
                    let field_coords = &coordinates[field_slot];
                    (
                        galerkin_quad4_weak_pair(
                            source,
                            field,
                            source_coords,
                            field_coords,
                            exterior,
                            local_basis_count,
                            gauss_order,
                        ),
                        galerkin_quad4_weak_pair(
                            source,
                            field,
                            source_coords,
                            field_coords,
                            interior,
                            local_basis_count,
                            gauss_order,
                        ),
                    )
                });
            let exterior_w = if is_closed {
                -exterior_pair.curl_integrals[source_local][field_local]
                    + Complex64::new(exterior.wave_number * exterior.wave_number, 0.0)
                        * exterior_pair.normal_mass_integrals[source_local][field_local]
            } else {
                exterior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            let interior_w = if is_closed {
                -interior_pair.curl_integrals[source_local][field_local]
                    + Complex64::new(interior.wave_number * interior.wave_number, 0.0)
                        * interior_pair.normal_mass_integrals[source_local][field_local]
            } else {
                interior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            value += exterior_w / exterior.density + interior_w / interior.density;
        }
    }
    value
}

fn galerkin_muller_entry(
    row: usize,
    column: usize,
    interface_dofs: usize,
    incidence: &[Vec<(usize, usize)>],
    elements: &[Element],
    active: &[usize],
    coordinates: &[DenseMatrix<f64>],
    geometry: &[AffineTriangleGeometry],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    is_closed: bool,
) -> Complex64 {
    let row_block = usize::from(row >= interface_dofs);
    let column_block = usize::from(column >= interface_dofs);
    let row_dof = row % interface_dofs;
    let column_dof = column % interface_dofs;
    let local_basis_count = match basis {
        BasisOrder::Linear => 3,
        BasisOrder::Quadratic => 6,
        BasisOrder::Constant => unreachable!(),
    };
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let mut value = Complex64::new(0.0, 0.0);

    for &(source_slot, source_local) in &incidence[row_dof] {
        for &(field_slot, field_local) in &incidence[column_dof] {
            debug_assert!(source_local < local_basis_count && field_local < local_basis_count);
            let source = &elements[active[source_slot]];
            let field = &elements[active[field_slot]];
            let source_coords = &coordinates[source_slot];
            let field_coords = &coordinates[field_slot];

            let exterior_pair = galerkin_pair_with_basis(
                source,
                field,
                source_coords,
                field_coords,
                exterior,
                local_basis_count,
                gauss_order,
            );
            let interior_pair = galerkin_pair_with_basis(
                source,
                field,
                source_coords,
                field_coords,
                interior,
                local_basis_count,
                gauss_order,
            );
            let exterior_hypersingular = if is_closed {
                galerkin_hypersingular_entry(
                    source,
                    field,
                    source_coords,
                    field_coords,
                    geometry[source_slot],
                    geometry[field_slot],
                    exterior,
                    basis,
                    source_local,
                    field_local,
                    gauss_order,
                    &exterior_pair,
                )
            } else {
                exterior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            let interior_hypersingular = if is_closed {
                galerkin_hypersingular_entry(
                    source,
                    field,
                    source_coords,
                    field_coords,
                    geometry[source_slot],
                    geometry[field_slot],
                    interior,
                    basis,
                    source_local,
                    field_local,
                    gauss_order,
                    &interior_pair,
                )
            } else {
                interior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            let exterior_entry = &exterior_pair.basis_results[source_local][field_local];
            let interior_entry = &interior_pair.basis_results[source_local][field_local];
            value += match (row_block, column_block) {
                (0, 0) => exterior_entry.dg_dn_integral + interior_entry.dg_dn_integral,
                (0, 1) => {
                    -i_omega
                        * (exterior.density * exterior_entry.g_integral
                            + interior.density * interior_entry.g_integral)
                }
                (1, 0) => {
                    exterior_hypersingular / exterior.density
                        + interior_hypersingular / interior.density
                }
                (1, 1) => {
                    -i_omega * (exterior_entry.dg_dnx_integral + interior_entry.dg_dnx_integral)
                }
                _ => unreachable!(),
            };
        }
    }
    if row_block == 1 && column_block == 0 && row_dof != column_dof {
        let reverse = galerkin_hypersingular_global_entry(
            &incidence[column_dof],
            &incidence[row_dof],
            elements,
            active,
            coordinates,
            geometry,
            exterior,
            interior,
            basis,
            gauss_order,
            is_closed,
        );
        value = (value + reverse) * 0.5;
    }
    value
}

fn galerkin_hypersingular_global_entry(
    source_incidence: &[(usize, usize)],
    field_incidence: &[(usize, usize)],
    elements: &[Element],
    active: &[usize],
    coordinates: &[DenseMatrix<f64>],
    geometry: &[AffineTriangleGeometry],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    is_closed: bool,
) -> Complex64 {
    let local_basis_count = match basis {
        BasisOrder::Linear => 3,
        BasisOrder::Quadratic => 6,
        BasisOrder::Constant => unreachable!(),
    };
    let mut value = Complex64::new(0.0, 0.0);
    for &(source_slot, source_local) in source_incidence {
        for &(field_slot, field_local) in field_incidence {
            let source = &elements[active[source_slot]];
            let field = &elements[active[field_slot]];
            let source_coords = &coordinates[source_slot];
            let field_coords = &coordinates[field_slot];
            let exterior_pair = galerkin_pair_with_basis(
                source,
                field,
                source_coords,
                field_coords,
                exterior,
                local_basis_count,
                gauss_order,
            );
            let interior_pair = galerkin_pair_with_basis(
                source,
                field,
                source_coords,
                field_coords,
                interior,
                local_basis_count,
                gauss_order,
            );
            let exterior_hypersingular = if is_closed {
                galerkin_hypersingular_entry(
                    source,
                    field,
                    source_coords,
                    field_coords,
                    geometry[source_slot],
                    geometry[field_slot],
                    exterior,
                    basis,
                    source_local,
                    field_local,
                    gauss_order,
                    &exterior_pair,
                )
            } else {
                exterior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            let interior_hypersingular = if is_closed {
                galerkin_hypersingular_entry(
                    source,
                    field,
                    source_coords,
                    field_coords,
                    geometry[source_slot],
                    geometry[field_slot],
                    interior,
                    basis,
                    source_local,
                    field_local,
                    gauss_order,
                    &interior_pair,
                )
            } else {
                interior_pair.basis_results[source_local][field_local].d2g_dnxdny_integral
            };
            value += exterior_hypersingular / exterior.density
                + interior_hypersingular / interior.density;
        }
    }
    value
}

fn galerkin_hypersingular_entry(
    source: &Element,
    field: &Element,
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_geometry: AffineTriangleGeometry,
    field_geometry: AffineTriangleGeometry,
    physics: &PhysicsParams,
    basis: BasisOrder,
    source_local: usize,
    field_local: usize,
    gauss_order: usize,
    mass_result: &GalerkinIntegrationResult,
) -> Complex64 {
    let curl_integral = if basis == BasisOrder::Quadratic {
        let derivative_result = galerkin_linear_single_layer_pair(
            source,
            field,
            source_coords,
            field_coords,
            physics,
            gauss_order,
        );
        quadratic_curl_integral(
            &derivative_result,
            quadratic_curl_coefficients(source_geometry.curls)[source_local],
            quadratic_curl_coefficients(field_geometry.curls)[field_local],
        )
    } else {
        let source_curl = source_geometry.curls[source_local];
        let field_curl = field_geometry.curls[field_local];
        let curl_dot = source_curl[0] * field_curl[0]
            + source_curl[1] * field_curl[1]
            + source_curl[2] * field_curl[2];
        let constant_single_layer = mass_result
            .basis_results
            .iter()
            .flat_map(|row| row.iter())
            .map(|entry| entry.g_integral)
            .sum::<Complex64>();
        Complex64::new(curl_dot, 0.0) * constant_single_layer
    };
    let normal_dot = source_geometry.normal[0] * field_geometry.normal[0]
        + source_geometry.normal[1] * field_geometry.normal[1]
        + source_geometry.normal[2] * field_geometry.normal[2];
    -curl_integral
        + Complex64::new(physics.wave_number * physics.wave_number * normal_dot, 0.0)
            * mass_result.basis_results[source_local][field_local].g_integral
}

/// Assemble the affine-Tri3 Galerkin hypersingular operator for conforming
/// P1 or P2 surfaces.
///
/// For continuous affine traces on a closed surface, integration by parts
/// converts the finite-part kernel into the weakly singular expression
///
/// `W(u,v) = ∬ G (k² (n_x · n_y) u v - curlΓu · curlΓv) dS_y dS_x`.
///
/// The implementation therefore reuses the production Duffy and
/// Sauter–Schwab single-layer charts for self, touching, and separated panel
/// pairs. Open surfaces use the finite-part `d²G/(∂n_x∂n_y)` channel from those
/// charts, retaining boundary-edge contributions omitted by the closed weak
/// form. Both paths validate shared vertex/edge DOFs and winding. Quadratic
/// traces use the affine Tri6 basis and its linear barycentric derivatives.
pub fn assemble_galerkin_hypersingular_with_global_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
) -> Result<GalerkinHypersingularOperators, GalerkinAssemblyError> {
    if elements.iter().any(|element| {
        !element.property.is_evaluation() && element.element_type == ElementType::Quad4
    }) {
        return assemble_galerkin_hypersingular_quad4_global(
            elements,
            nodes,
            physics,
            basis,
            gauss_order,
            global_dof_count,
        );
    }
    let local_basis_count = match basis {
        BasisOrder::Linear => 3,
        BasisOrder::Quadratic => 6,
        BasisOrder::Constant => {
            return Err(GalerkinAssemblyError::UnsupportedHypersingularBasis { basis });
        }
    };
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let active = match basis {
        BasisOrder::Linear => {
            validate_continuous_open_linear_layout(elements, nodes, global_dof_count)?
        }
        BasisOrder::Quadratic => {
            validate_continuous_open_quadratic_layout(elements, nodes, global_dof_count)?
        }
        BasisOrder::Constant => unreachable!("constant hypersingular basis rejected above"),
    };
    let is_closed = triangular_hypersingular_surface_is_closed(elements, &active);
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let geometry = active
        .iter()
        .enumerate()
        .map(|(active_index, &element_index)| {
            affine_triangle_geometry(&coordinates[active_index], element_index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut matrix = DenseMatrix::zeros(global_dof_count, global_dof_count);
    let k2 = physics.wave_number * physics.wave_number;

    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let mass_result = galerkin_pair_with_basis(
                &elements[source_index],
                &elements[field_index],
                &coordinates[row_panel],
                &coordinates[column_panel],
                physics,
                local_basis_count,
                gauss_order,
            );
            let derivative_result = if is_closed && basis == BasisOrder::Quadratic {
                Some(galerkin_linear_single_layer_pair(
                    &elements[source_index],
                    &elements[field_index],
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    gauss_order,
                ))
            } else {
                None
            };
            let normal_dot = geometry[row_panel].normal[0] * geometry[column_panel].normal[0]
                + geometry[row_panel].normal[1] * geometry[column_panel].normal[1]
                + geometry[row_panel].normal[2] * geometry[column_panel].normal[2];
            let quadratic_curls = if basis == BasisOrder::Quadratic {
                Some((
                    quadratic_curl_coefficients(geometry[row_panel].curls),
                    quadratic_curl_coefficients(geometry[column_panel].curls),
                ))
            } else {
                None
            };

            for local_row in 0..local_basis_count {
                let row = elements[source_index].dof_addresses[local_row];
                for local_column in 0..local_basis_count {
                    let column = elements[field_index].dof_addresses[local_column];
                    let value = if !is_closed {
                        mass_result.basis_results[local_row][local_column].d2g_dnxdny_integral
                    } else {
                        let curl_integral = if let Some(derivative_result) = &derivative_result {
                            let (source_coefficients, field_coefficients) =
                                quadratic_curls.expect("quadratic derivative coefficients");
                            quadratic_curl_integral(
                                derivative_result,
                                source_coefficients[local_row],
                                field_coefficients[local_column],
                            )
                        } else {
                            let result = &mass_result;
                            let constant_single_layer = result
                                .basis_results
                                .iter()
                                .flat_map(|row| row.iter())
                                .map(|entry| entry.g_integral)
                                .sum::<Complex64>();
                            let source_curl = geometry[row_panel].curls[local_row];
                            let field_curl = geometry[column_panel].curls[local_column];
                            let curl_dot = source_curl[0] * field_curl[0]
                                + source_curl[1] * field_curl[1]
                                + source_curl[2] * field_curl[2];
                            Complex64::new(curl_dot, 0.0) * constant_single_layer
                        };
                        -curl_integral
                            + Complex64::new(k2 * normal_dot, 0.0)
                                * mass_result.basis_results[local_row][local_column].g_integral
                    };
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    matrix[[row, column]] += value;
                }
            }
        }
    }

    // The weak Helmholtz form is complex-symmetric (reciprocal), rather than
    // Hermitian.  The touching-panel charts use different but equivalent
    // Sauter–Schwab orderings in the two directions, so average their tiny
    // quadrature-order asymmetry without conjugating the entries.
    for row in 0..global_dof_count {
        for column in (row + 1)..global_dof_count {
            let value = (matrix[[row, column]] + matrix[[column, row]]) * 0.5;
            matrix[[row, column]] = value;
            matrix[[column, row]] = value;
        }
    }

    Ok(GalerkinHypersingularOperators {
        hypersingular: matrix,
    })
}

fn galerkin_pair_with_basis(
    source: &Element,
    field: &Element,
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    basis_count: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    match detect_adjacency(source, field) {
        AdjacencyType::SelfElement => galerkin_single_layer_self_triangle_with_basis(
            source_coords,
            physics,
            basis_count,
            basis_count,
            gauss_order,
        ),
        AdjacencyType::CommonEdge { .. } => sauter_schwab_galerkin_common_edge(
            source_coords,
            field_coords,
            physics,
            basis_count,
            basis_count,
            gauss_order,
        ),
        AdjacencyType::CommonVertex { .. } => sauter_schwab_galerkin_common_vertex(
            source_coords,
            field_coords,
            physics,
            basis_count,
            basis_count,
            gauss_order,
        ),
        AdjacencyType::Separated => galerkin_single_layer_regular_triangle_with_basis(
            source_coords,
            field_coords,
            physics,
            basis_count,
            basis_count,
            gauss_order,
        ),
    }
}

fn galerkin_linear_single_layer_pair(
    source: &Element,
    field: &Element,
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    galerkin_pair_with_basis(
        source,
        field,
        source_coords,
        field_coords,
        physics,
        3,
        gauss_order,
    )
}

fn quadratic_curl_coefficients(curls: [[f64; 3]; 3]) -> [[[f64; 3]; 3]; 6] {
    [
        [
            [3.0 * curls[0][0], 3.0 * curls[0][1], 3.0 * curls[0][2]],
            [-curls[0][0], -curls[0][1], -curls[0][2]],
            [-curls[0][0], -curls[0][1], -curls[0][2]],
        ],
        [
            [-curls[1][0], -curls[1][1], -curls[1][2]],
            [3.0 * curls[1][0], 3.0 * curls[1][1], 3.0 * curls[1][2]],
            [-curls[1][0], -curls[1][1], -curls[1][2]],
        ],
        [
            [-curls[2][0], -curls[2][1], -curls[2][2]],
            [-curls[2][0], -curls[2][1], -curls[2][2]],
            [3.0 * curls[2][0], 3.0 * curls[2][1], 3.0 * curls[2][2]],
        ],
        [
            [4.0 * curls[1][0], 4.0 * curls[1][1], 4.0 * curls[1][2]],
            [4.0 * curls[0][0], 4.0 * curls[0][1], 4.0 * curls[0][2]],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [4.0 * curls[2][0], 4.0 * curls[2][1], 4.0 * curls[2][2]],
            [4.0 * curls[1][0], 4.0 * curls[1][1], 4.0 * curls[1][2]],
        ],
        [
            [4.0 * curls[2][0], 4.0 * curls[2][1], 4.0 * curls[2][2]],
            [0.0, 0.0, 0.0],
            [4.0 * curls[0][0], 4.0 * curls[0][1], 4.0 * curls[0][2]],
        ],
    ]
}

fn quadratic_curl_integral(
    linear_result: &GalerkinIntegrationResult,
    source_coefficients: [[f64; 3]; 3],
    field_coefficients: [[f64; 3]; 3],
) -> Complex64 {
    let mut integral = Complex64::new(0.0, 0.0);
    for (source_coefficients, basis_row) in source_coefficients
        .iter()
        .zip(linear_result.basis_results.iter())
    {
        for (field_coefficients, basis_result) in field_coefficients.iter().zip(basis_row.iter()) {
            let dot = source_coefficients[0] * field_coefficients[0]
                + source_coefficients[1] * field_coefficients[1]
                + source_coefficients[2] * field_coefficients[2];
            integral += Complex64::new(dot, 0.0) * basis_result.g_integral;
        }
    }
    integral
}

#[derive(Debug, Clone, Copy)]
struct AffineTriangleGeometry {
    normal: [f64; 3],
    curls: [[f64; 3]; 3],
}

fn affine_triangle_geometry(
    coords: &DenseMatrix<f64>,
    element: usize,
) -> Result<AffineTriangleGeometry, GalerkinAssemblyError> {
    let p0 = [coords[[0, 0]], coords[[0, 1]], coords[[0, 2]]];
    let p1 = [coords[[1, 0]], coords[[1, 1]], coords[[1, 2]]];
    let p2 = [coords[[2, 0]], coords[[2, 1]], coords[[2, 2]]];
    let edge_01 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let edge_02 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        edge_01[1] * edge_02[2] - edge_01[2] * edge_02[1],
        edge_01[2] * edge_02[0] - edge_01[0] * edge_02[2],
        edge_01[0] * edge_02[1] - edge_01[1] * edge_02[0],
    ];
    let jacobian = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if !jacobian.is_finite() || jacobian <= 1.0e-15 {
        return Err(GalerkinAssemblyError::DegenerateHypersingularElement { element });
    }
    let normal = [
        cross[0] / jacobian,
        cross[1] / jacobian,
        cross[2] / jacobian,
    ];
    let cross_with_normal = |vector: [f64; 3]| {
        [
            normal[1] * vector[2] - normal[2] * vector[1],
            normal[2] * vector[0] - normal[0] * vector[2],
            normal[0] * vector[1] - normal[1] * vector[0],
        ]
    };
    let grad_0 = cross_with_normal([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]);
    let grad_1 = cross_with_normal([p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]]);
    let grad_2 = cross_with_normal([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]);
    let gradients = [
        [
            grad_0[0] / jacobian,
            grad_0[1] / jacobian,
            grad_0[2] / jacobian,
        ],
        [
            grad_1[0] / jacobian,
            grad_1[1] / jacobian,
            grad_1[2] / jacobian,
        ],
        [
            grad_2[0] / jacobian,
            grad_2[1] / jacobian,
            grad_2[2] / jacobian,
        ],
    ];
    let curls = gradients.map(|gradient| {
        [
            normal[1] * gradient[2] - normal[2] * gradient[1],
            normal[2] * gradient[0] - normal[0] * gradient[2],
            normal[0] * gradient[1] - normal[1] * gradient[0],
        ]
    });
    Ok(AffineTriangleGeometry { normal, curls })
}

fn validate_continuous_open_linear_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    global_dof_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    validate_continuous_linear_layout(elements, nodes, global_dof_count, false)
}

fn validate_continuous_linear_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    global_dof_count: usize,
    require_closed: bool,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let active = active_triangular_global_layout(elements, nodes, 3, global_dof_count)?;
    let mut node_dofs = HashMap::<usize, usize>::new();
    let mut dof_nodes = HashMap::<usize, usize>::new();
    let mut edge_incidence = HashMap::<(usize, usize), usize>::new();

    for &element_index in &active {
        let element = &elements[element_index];
        if element.connectivity.len() != 3 {
            return Err(GalerkinAssemblyError::InvalidHypersingularConnectivity {
                element: element_index,
                nodes: element.connectivity.len(),
            });
        }
        for local in 0..3 {
            let node = element.connectivity[local];
            let dof = element.dof_addresses[local];
            if let Some(&expected) = node_dofs.get(&node) {
                if expected != dof {
                    return Err(GalerkinAssemblyError::NonConformingHypersingularDof {
                        element: element_index,
                        local,
                        node,
                        actual: dof,
                        expected,
                    });
                }
            } else {
                node_dofs.insert(node, dof);
            }
            if let Some(&expected_node) = dof_nodes.get(&dof) {
                if expected_node != node {
                    let expected = node_dofs.get(&expected_node).copied().unwrap_or(dof);
                    return Err(GalerkinAssemblyError::NonConformingHypersingularDof {
                        element: element_index,
                        local,
                        node,
                        actual: dof,
                        expected,
                    });
                }
            } else {
                dof_nodes.insert(dof, node);
            }
        }
        for edge in [
            (element.connectivity[0], element.connectivity[1]),
            (element.connectivity[1], element.connectivity[2]),
            (element.connectivity[2], element.connectivity[0]),
        ] {
            let key = if edge.0 < edge.1 {
                edge
            } else {
                (edge.1, edge.0)
            };
            *edge_incidence.entry(key).or_default() += 1;
        }
    }
    if require_closed
        && let Some((&(first, second), &incidence)) = edge_incidence
            .iter()
            .find(|&(_, &incidence)| incidence != 2)
    {
        return Err(GalerkinAssemblyError::OpenHypersingularSurface {
            first,
            second,
            incidence,
        });
    }
    validate_hypersingular_orientation(elements, &active)?;
    Ok(active)
}

/// Ensure each shared edge is traversed in opposite directions by its two
/// incident triangles.  The weak hypersingular integration-by-parts identity
/// assumes a consistently oriented closed two-manifold; accepting same-way
/// winding would silently flip one panel's conormal contribution.
fn validate_hypersingular_orientation(
    elements: &[Element],
    active: &[usize],
) -> Result<(), GalerkinAssemblyError> {
    let mut directed_edges = HashMap::<(usize, usize), (usize, usize, usize, usize)>::new();
    for &element_index in active {
        let element = &elements[element_index];
        for (local, (start, end)) in [
            (element.connectivity[0], element.connectivity[1]),
            (element.connectivity[1], element.connectivity[2]),
            (element.connectivity[2], element.connectivity[0]),
        ]
        .into_iter()
        .enumerate()
        {
            let key = if start < end {
                (start, end)
            } else {
                (end, start)
            };
            if let Some(&(first_element, _, first_start, first_end)) = directed_edges.get(&key) {
                if first_start == start && first_end == end {
                    return Err(
                        GalerkinAssemblyError::InconsistentHypersingularOrientation {
                            first: key.0,
                            second: key.1,
                            first_element,
                            second_element: element_index,
                        },
                    );
                }
            } else {
                directed_edges.insert(key, (element_index, local, start, end));
            }
        }
    }
    Ok(())
}

/// Ensure each shared edge of a closed Quad4 surface is traversed in
/// opposite directions by its two incident panels. The weak-form
/// hypersingular identity has the same conormal-cancellation requirement for
/// quadrilateral panels as it does for Tri3 panels.
fn validate_quad4_hypersingular_orientation(
    elements: &[Element],
    active: &[usize],
) -> Result<(), GalerkinAssemblyError> {
    let mut directed_edges = HashMap::<(usize, usize), (usize, usize, usize)>::new();

    for &element_index in active {
        let element = &elements[element_index];
        if element.connectivity.len() != 4 {
            return Err(GalerkinAssemblyError::InvalidQuad4Connectivity {
                element: element_index,
                nodes: element.connectivity.len(),
            });
        }

        for local in 0..4 {
            let start = element.connectivity[local];
            let end = element.connectivity[(local + 1) % 4];
            let key = if start < end {
                (start, end)
            } else {
                (end, start)
            };

            if let Some(&(first_start, first_end, first_element)) = directed_edges.get(&key) {
                if (start, end) == (first_start, first_end) {
                    return Err(
                        GalerkinAssemblyError::InconsistentHypersingularOrientation {
                            first: key.0,
                            second: key.1,
                            first_element,
                            second_element: element_index,
                        },
                    );
                }
            } else {
                directed_edges.insert(key, (start, end, element_index));
            }
        }
    }

    Ok(())
}

fn triangular_hypersingular_surface_is_closed(elements: &[Element], active: &[usize]) -> bool {
    let mut edge_incidence = HashMap::<(usize, usize), usize>::new();
    for &element_index in active {
        let connectivity = &elements[element_index].connectivity;
        for edge in [
            (connectivity[0], connectivity[1]),
            (connectivity[1], connectivity[2]),
            (connectivity[2], connectivity[0]),
        ] {
            let key = if edge.0 < edge.1 {
                edge
            } else {
                (edge.1, edge.0)
            };
            *edge_incidence.entry(key).or_default() += 1;
        }
    }
    edge_incidence.values().all(|&incidence| incidence == 2)
}

fn quad4_hypersingular_surface_is_closed(elements: &[Element], active: &[usize]) -> bool {
    let mut edge_incidence = HashMap::<(usize, usize), usize>::new();
    for &element_index in active {
        let connectivity = &elements[element_index].connectivity;
        for edge in 0..4 {
            let first = connectivity[edge];
            let second = connectivity[(edge + 1) % 4];
            let key = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            *edge_incidence.entry(key).or_default() += 1;
        }
    }
    edge_incidence.values().all(|&incidence| incidence == 2)
}

fn validate_continuous_open_quadratic_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    global_dof_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    validate_continuous_quadratic_layout(elements, nodes, global_dof_count, false)
}

fn validate_continuous_quadratic_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    global_dof_count: usize,
    require_closed: bool,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let active = active_triangular_global_layout(elements, nodes, 6, global_dof_count)?;
    let mut node_dofs = HashMap::<usize, usize>::new();
    let mut dof_nodes = HashMap::<usize, usize>::new();
    let mut edge_dofs = HashMap::<(usize, usize), usize>::new();
    let mut dof_edges = HashMap::<usize, (usize, usize)>::new();
    let mut edge_incidence = HashMap::<(usize, usize), usize>::new();

    for &element_index in &active {
        let element = &elements[element_index];
        if element.connectivity.len() != 3 {
            return Err(GalerkinAssemblyError::InvalidHypersingularConnectivity {
                element: element_index,
                nodes: element.connectivity.len(),
            });
        }
        for local in 0..3 {
            let node = element.connectivity[local];
            let dof = element.dof_addresses[local];
            if let Some(&expected) = node_dofs.get(&node) {
                if expected != dof {
                    return Err(GalerkinAssemblyError::NonConformingHypersingularDof {
                        element: element_index,
                        local,
                        node,
                        actual: dof,
                        expected,
                    });
                }
            } else {
                node_dofs.insert(node, dof);
            }
            if dof_edges.contains_key(&dof) {
                return Err(
                    GalerkinAssemblyError::HypersingularGlobalDofEntityConflict {
                        element: element_index,
                        local,
                        dof,
                    },
                );
            }
            if let Some(&expected_node) = dof_nodes.get(&dof) {
                if expected_node != node {
                    return Err(GalerkinAssemblyError::NonConformingHypersingularDof {
                        element: element_index,
                        local,
                        node,
                        actual: dof,
                        expected: node_dofs.get(&expected_node).copied().unwrap_or(dof),
                    });
                }
            } else {
                dof_nodes.insert(dof, node);
            }
        }

        let edges = [
            (element.connectivity[0], element.connectivity[1]),
            (element.connectivity[1], element.connectivity[2]),
            (element.connectivity[2], element.connectivity[0]),
        ];
        for (edge_local, edge) in edges.into_iter().enumerate() {
            let key = if edge.0 < edge.1 {
                edge
            } else {
                (edge.1, edge.0)
            };
            let local = 3 + edge_local;
            let dof = element.dof_addresses[local];
            if let Some(&expected) = edge_dofs.get(&key) {
                if expected != dof {
                    return Err(GalerkinAssemblyError::NonConformingHypersingularEdgeDof {
                        element: element_index,
                        first: key.0,
                        second: key.1,
                        actual: dof,
                        expected,
                    });
                }
            } else {
                edge_dofs.insert(key, dof);
            }
            if dof_nodes.contains_key(&dof)
                || dof_edges.get(&dof).is_some_and(|expected| *expected != key)
            {
                return Err(
                    GalerkinAssemblyError::HypersingularGlobalDofEntityConflict {
                        element: element_index,
                        local,
                        dof,
                    },
                );
            }
            dof_edges.insert(dof, key);
            *edge_incidence.entry(key).or_default() += 1;
        }
    }
    if require_closed
        && let Some((&(first, second), &incidence)) = edge_incidence
            .iter()
            .find(|&(_, &incidence)| incidence != 2)
    {
        return Err(GalerkinAssemblyError::OpenHypersingularSurface {
            first,
            second,
            incidence,
        });
    }
    validate_hypersingular_orientation(elements, &active)?;
    Ok(active)
}

fn active_triangular_linear_layout(
    elements: &[Element],
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    active_triangular_basis_layout(elements, 3)
}

fn active_triangular_quadratic_layout(
    elements: &[Element],
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    active_triangular_basis_layout(elements, 6)
}

fn active_triangular_basis_layout(
    elements: &[Element],
    local_basis_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let mut active = Vec::new();
    for (element, value) in elements.iter().enumerate() {
        if value.property.is_evaluation() {
            continue;
        }
        if value.element_type != ElementType::Tri3 {
            return Err(GalerkinAssemblyError::UnsupportedElementType {
                element,
                element_type: value.element_type,
            });
        }
        if value.dof_addresses.len() != local_basis_count {
            return Err(GalerkinAssemblyError::InvalidBasisDofs {
                element,
                dofs: value.dof_addresses.len(),
            });
        }
        let expected_base = active.len() * local_basis_count;
        for (local, &actual) in value.dof_addresses.iter().enumerate() {
            let expected = expected_base + local;
            if actual != expected {
                return Err(GalerkinAssemblyError::DofLayout {
                    element,
                    actual,
                    expected,
                });
            }
        }
        active.push(element);
    }
    if active.is_empty() {
        return Err(GalerkinAssemblyError::EmptySurface);
    }
    Ok(active)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use sonium_solvers::dense::DenseMatrix;

    use super::{dense_relative_residual, quadratic_curl_coefficients};

    #[test]
    fn dense_residual_does_not_hide_tiny_nonzero_rhs() {
        let mut matrix = DenseMatrix::zeros(1, 1);
        matrix[[0, 0]] = Complex64::new(1.0, 0.0);
        let rhs = [Complex64::new(1.0e-100, 0.0)];
        let solution = [Complex64::new(0.0, 0.0)];
        let residual = dense_relative_residual(&matrix, &solution, &rhs);
        assert!(
            residual > 0.5,
            "tiny RHS was incorrectly scaled: {residual:e}"
        );
    }

    #[test]
    fn quadratic_curl_coefficients_match_tri6_derivatives() {
        let curls = [[1.0, -2.0, 0.5], [-0.25, 3.0, 1.5], [2.0, 0.75, -1.0]];
        let coefficients = quadratic_curl_coefficients(curls);
        let lambda = [0.2, 0.3, 0.5];
        let evaluate = |basis: usize| {
            (0..3)
                .map(|component| {
                    (0..3)
                        .map(|linear| coefficients[basis][linear][component] * lambda[linear])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        };
        let expected = [
            [
                (4.0 * lambda[0] - 1.0) * curls[0][0],
                (4.0 * lambda[0] - 1.0) * curls[0][1],
                (4.0 * lambda[0] - 1.0) * curls[0][2],
            ],
            [
                (4.0 * lambda[1] - 1.0) * curls[1][0],
                (4.0 * lambda[1] - 1.0) * curls[1][1],
                (4.0 * lambda[1] - 1.0) * curls[1][2],
            ],
            [
                (4.0 * lambda[2] - 1.0) * curls[2][0],
                (4.0 * lambda[2] - 1.0) * curls[2][1],
                (4.0 * lambda[2] - 1.0) * curls[2][2],
            ],
            [
                4.0 * (lambda[1] * curls[0][0] + lambda[0] * curls[1][0]),
                4.0 * (lambda[1] * curls[0][1] + lambda[0] * curls[1][1]),
                4.0 * (lambda[1] * curls[0][2] + lambda[0] * curls[1][2]),
            ],
            [
                4.0 * (lambda[2] * curls[1][0] + lambda[1] * curls[2][0]),
                4.0 * (lambda[2] * curls[1][1] + lambda[1] * curls[2][1]),
                4.0 * (lambda[2] * curls[1][2] + lambda[1] * curls[2][2]),
            ],
            [
                4.0 * (lambda[0] * curls[2][0] + lambda[2] * curls[0][0]),
                4.0 * (lambda[0] * curls[2][1] + lambda[2] * curls[0][1]),
                4.0 * (lambda[0] * curls[2][2] + lambda[2] * curls[0][2]),
            ],
        ];
        for basis in 0..6 {
            let actual = evaluate(basis);
            for component in 0..3 {
                assert!((actual[component] - expected[basis][component]).abs() < 1.0e-12);
            }
        }
    }
}

#[test]
fn galerkin_transmission_batch_solves_match_scalar_forward_and_adjoint() {
    let mut matrix = DenseMatrix::zeros(4, 4);
    matrix[[0, 0]] = Complex64::new(4.0, 0.5);
    matrix[[0, 1]] = Complex64::new(0.2, -0.1);
    matrix[[0, 2]] = Complex64::new(-0.3, 0.2);
    matrix[[0, 3]] = Complex64::new(0.1, 0.0);
    matrix[[1, 0]] = Complex64::new(-0.4, 0.1);
    matrix[[1, 1]] = Complex64::new(3.0, -0.2);
    matrix[[1, 2]] = Complex64::new(0.25, 0.3);
    matrix[[1, 3]] = Complex64::new(-0.15, 0.05);
    matrix[[2, 0]] = Complex64::new(0.1, 0.0);
    matrix[[2, 1]] = Complex64::new(-0.2, 0.4);
    matrix[[2, 2]] = Complex64::new(2.5, 0.1);
    matrix[[2, 3]] = Complex64::new(0.3, -0.2);
    matrix[[3, 0]] = Complex64::new(0.05, -0.1);
    matrix[[3, 1]] = Complex64::new(0.2, 0.0);
    matrix[[3, 2]] = Complex64::new(-0.1, 0.25);
    matrix[[3, 3]] = Complex64::new(2.0, -0.3);
    let operators = super::GalerkinTransmissionOperators {
        matrix,
        incident_operator: DenseMatrix::zeros(4, 4),
        interface_dofs: 2,
        basis_order: crate::core::types::BasisOrder::Linear,
    };
    let rhs_columns = vec![
        vec![
            Complex64::new(1.0, -0.2),
            Complex64::new(-0.3, 0.4),
            Complex64::new(0.5, 0.1),
            Complex64::new(-0.2, -0.6),
        ],
        vec![
            Complex64::new(-0.2, 0.7),
            Complex64::new(0.8, -0.1),
            Complex64::new(-0.4, 0.3),
            Complex64::new(0.6, 0.2),
        ],
    ];
    let forward_batch = operators
        .solve_many(&rhs_columns, 1.0e-12)
        .expect("forward batch solve");
    let adjoint_batch = operators
        .solve_adjoint_many(&rhs_columns, 1.0e-12)
        .expect("adjoint batch solve");
    assert_eq!(forward_batch.len(), rhs_columns.len());
    assert_eq!(adjoint_batch.len(), rhs_columns.len());
    for (index, rhs) in rhs_columns.iter().enumerate() {
        let forward_scalar = operators.solve(rhs, 1.0e-12).expect("forward scalar solve");
        let adjoint_scalar = operators
            .solve_adjoint(rhs, 1.0e-12)
            .expect("adjoint scalar solve");
        for row in 0..operators.total_dofs() {
            assert!((forward_batch[index][row] - forward_scalar[row]).norm() < 1.0e-12);
            assert!((adjoint_batch[index][row] - adjoint_scalar[row]).norm() < 1.0e-12);
        }
    }
}

#[test]
fn galerkin_transmission_batch_validates_empty_tolerance_and_rhs() {
    let operators = super::GalerkinTransmissionOperators {
        matrix: {
            let mut matrix = DenseMatrix::zeros(2, 2);
            matrix[[0, 0]] = Complex64::new(2.0, 0.0);
            matrix[[1, 1]] = Complex64::new(3.0, 0.0);
            matrix
        },
        incident_operator: DenseMatrix::zeros(2, 2),
        interface_dofs: 1,
        basis_order: crate::core::types::BasisOrder::Linear,
    };
    assert!(matches!(
        operators.solve_many(&[], f64::NAN),
        Err(super::GalerkinTransmissionSolveError::InvalidTolerance { .. })
    ));
    assert!(matches!(
        operators.solve_adjoint_many(
            &[vec![Complex64::new(f64::NAN, 0.0), Complex64::default()]],
            1.0e-8
        ),
        Err(super::GalerkinTransmissionSolveError::NonFiniteRhs { index: 0 })
    ));
    assert!(matches!(
        operators.solve_many(&[vec![Complex64::default()]], 1.0e-8),
        Err(super::GalerkinTransmissionSolveError::RhsLength {
            expected: 2,
            actual: 1
        })
    ));
}

fn galerkin_quad4_pair(
    source: &Element,
    field: &Element,
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    basis_count: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    match detect_adjacency(source, field) {
        AdjacencyType::SelfElement => galerkin_single_layer_self_quad4_with_basis(
            source_coords,
            physics,
            basis_count,
            gauss_order,
        ),
        AdjacencyType::Separated => galerkin_single_layer_regular_quad4_with_basis(
            source_coords,
            field_coords,
            physics,
            basis_count,
            basis_count,
            gauss_order,
        ),
        AdjacencyType::CommonEdge { .. } | AdjacencyType::CommonVertex { .. } => {
            galerkin_single_layer_touching_quad4_with_basis(
                source_coords,
                field_coords,
                &source.connectivity,
                &field.connectivity,
                physics,
                basis_count,
                basis_count,
                gauss_order,
            )
        }
    }
}

fn galerkin_quad4_weak_pair(
    source: &Element,
    field: &Element,
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    basis_count: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    match detect_adjacency(source, field) {
        AdjacencyType::Separated => {
            let GalerkinQuad4WeakResult {
                mut kernels,
                curl_integrals,
                normal_mass_integrals,
            } = galerkin_quad4_regular_weak_form_with_basis(
                source_coords,
                field_coords,
                physics,
                basis_count,
                basis_count,
                gauss_order,
            );
            kernels.curl_integrals = curl_integrals;
            kernels.normal_mass_integrals = normal_mass_integrals;
            kernels
        }
        AdjacencyType::SelfElement => galerkin_single_layer_self_quad4_with_basis(
            source_coords,
            physics,
            basis_count,
            gauss_order,
        ),
        AdjacencyType::CommonEdge { .. } | AdjacencyType::CommonVertex { .. } => {
            galerkin_single_layer_touching_quad4_with_basis(
                source_coords,
                field_coords,
                &source.connectivity,
                &field.connectivity,
                physics,
                basis_count,
                basis_count,
                gauss_order,
            )
        }
    }
}

fn galerkin_quad4_weak_pair_wavenumber_derivative(
    source: &Element,
    field: &Element,
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    basis_count: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    match detect_adjacency(source, field) {
        AdjacencyType::Separated => {
            let GalerkinQuad4WeakResult {
                mut kernels,
                curl_integrals,
                normal_mass_integrals,
            } = galerkin_quad4_regular_weak_form_wavenumber_derivative_with_basis(
                source_coords,
                field_coords,
                physics,
                basis_count,
                basis_count,
                gauss_order,
            );
            kernels.curl_integrals = curl_integrals;
            kernels.normal_mass_integrals = normal_mass_integrals;
            kernels
        }
        AdjacencyType::SelfElement => {
            galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis(
                source_coords,
                physics,
                basis_count,
                gauss_order,
            )
        }
        AdjacencyType::CommonEdge { .. } | AdjacencyType::CommonVertex { .. } => {
            galerkin_single_layer_touching_quad4_wavenumber_derivative_with_basis(
                source_coords,
                field_coords,
                &source.connectivity,
                &field.connectivity,
                physics,
                basis_count,
                basis_count,
                gauss_order,
            )
        }
    }
}

fn assemble_galerkin_double_layer_quad4_global(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
) -> Result<GalerkinDoubleLayerOperators, GalerkinAssemblyError> {
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let basis_count = match basis {
        BasisOrder::Constant => 1,
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
    };
    let active =
        active_quadrilateral_global_layout(elements, nodes, basis_count, global_dof_count)?;
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let mut double_layer = DenseMatrix::zeros(global_dof_count, global_dof_count);
    let mut adjoint_double_layer = DenseMatrix::zeros(global_dof_count, global_dof_count);

    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let integrals = galerkin_quad4_pair(
                &elements[source_index],
                &elements[field_index],
                &coordinates[row_panel],
                &coordinates[column_panel],
                physics,
                basis_count,
                gauss_order,
            );
            for local_row in 0..basis_count {
                let row = elements[source_index].dof_addresses[local_row];
                for local_column in 0..basis_count {
                    let column = elements[field_index].dof_addresses[local_column];
                    let entry = &integrals.basis_results[local_row][local_column];
                    if !entry.dg_dn_integral.re.is_finite()
                        || !entry.dg_dn_integral.im.is_finite()
                        || !entry.dg_dnx_integral.re.is_finite()
                        || !entry.dg_dnx_integral.im.is_finite()
                    {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    double_layer[[row, column]] += entry.dg_dn_integral;
                    adjoint_double_layer[[row, column]] += entry.dg_dnx_integral;
                }
            }
        }
    }

    Ok(GalerkinDoubleLayerOperators {
        double_layer,
        adjoint_double_layer,
    })
}

fn assemble_galerkin_hypersingular_quad4_global(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
) -> Result<GalerkinHypersingularOperators, GalerkinAssemblyError> {
    if basis == BasisOrder::Constant {
        return Err(GalerkinAssemblyError::UnsupportedHypersingularBasis { basis });
    }
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let basis_count = match basis {
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
        BasisOrder::Constant => unreachable!("constant hypersingular basis rejected above"),
    };
    let active =
        validate_open_quad4_hypersingular_layout(elements, nodes, basis_count, global_dof_count)?;
    let is_closed = quad4_hypersingular_surface_is_closed(elements, &active);
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let mut matrix = DenseMatrix::zeros(global_dof_count, global_dof_count);
    let k2 = physics.wave_number * physics.wave_number;
    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let integrals = galerkin_quad4_weak_pair(
                &elements[source_index],
                &elements[field_index],
                &coordinates[row_panel],
                &coordinates[column_panel],
                physics,
                basis_count,
                gauss_order,
            );
            for local_row in 0..basis_count {
                let row = elements[source_index].dof_addresses[local_row];
                for local_column in 0..basis_count {
                    let column = elements[field_index].dof_addresses[local_column];
                    let value = if !is_closed {
                        integrals.basis_results[local_row][local_column].d2g_dnxdny_integral
                    } else {
                        -integrals.curl_integrals[local_row][local_column]
                            + Complex64::new(k2, 0.0)
                                * integrals.normal_mass_integrals[local_row][local_column]
                    };
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    matrix[[row, column]] += value;
                }
            }
        }
    }
    for row in 0..global_dof_count {
        for column in (row + 1)..global_dof_count {
            let value = (matrix[[row, column]] + matrix[[column, row]]) * 0.5;
            matrix[[row, column]] = value;
            matrix[[column, row]] = value;
        }
    }
    Ok(GalerkinHypersingularOperators {
        hypersingular: matrix,
    })
}

#[allow(dead_code)]
fn assemble_galerkin_single_layer_quad4_global(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    gauss_order: usize,
    global_dof_count: usize,
) -> Result<DenseMatrix<Complex64>, GalerkinAssemblyError> {
    if gauss_order == 0 {
        return Err(GalerkinAssemblyError::InvalidQuadratureOrder);
    }
    physics.validate()?;
    let local_basis_count = match basis {
        BasisOrder::Constant => 1,
        BasisOrder::Linear => 4,
        BasisOrder::Quadratic => 9,
    };
    let active =
        active_quadrilateral_global_layout(elements, nodes, local_basis_count, global_dof_count)?;
    let coordinates = active
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect::<Vec<_>>();
    let mut matrix = DenseMatrix::zeros(global_dof_count, global_dof_count);
    for (row_panel, &source_index) in active.iter().enumerate() {
        for (column_panel, &field_index) in active.iter().enumerate() {
            let adjacency = detect_adjacency(&elements[source_index], &elements[field_index]);
            let integrals = match adjacency {
                AdjacencyType::SelfElement => galerkin_single_layer_self_quad4_with_basis(
                    &coordinates[row_panel],
                    physics,
                    local_basis_count,
                    gauss_order,
                ),
                AdjacencyType::Separated => galerkin_single_layer_regular_quad4_with_basis(
                    &coordinates[row_panel],
                    &coordinates[column_panel],
                    physics,
                    local_basis_count,
                    local_basis_count,
                    gauss_order,
                ),
                AdjacencyType::CommonEdge { .. } | AdjacencyType::CommonVertex { .. } => {
                    galerkin_single_layer_touching_quad4_with_basis(
                        &coordinates[row_panel],
                        &coordinates[column_panel],
                        &elements[source_index].connectivity,
                        &elements[field_index].connectivity,
                        physics,
                        local_basis_count,
                        local_basis_count,
                        gauss_order,
                    )
                }
            };
            for local_row in 0..local_basis_count {
                let row = elements[source_index].dof_addresses[local_row];
                for local_column in 0..local_basis_count {
                    let column = elements[field_index].dof_addresses[local_column];
                    let value = integrals.basis_results[local_row][local_column].g_integral;
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(GalerkinAssemblyError::NonFiniteEntry { row, column });
                    }
                    matrix[[row, column]] += value;
                }
            }
        }
    }
    Ok(matrix)
}

#[allow(dead_code)]
fn active_quadrilateral_global_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    local_basis_count: usize,
    global_dof_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let mut active = Vec::new();
    for (element, value) in elements.iter().enumerate() {
        if value.property.is_evaluation() {
            continue;
        }
        if value.element_type != ElementType::Quad4 {
            return Err(GalerkinAssemblyError::UnsupportedElementType {
                element,
                element_type: value.element_type,
            });
        }
        if value.connectivity.len() != 4 {
            return Err(GalerkinAssemblyError::InvalidQuad4Connectivity {
                element,
                nodes: value.connectivity.len(),
            });
        }
        for (local, &node) in value.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(GalerkinAssemblyError::NodeOutOfRange {
                    element,
                    local,
                    node,
                    node_count: nodes.len(),
                });
            }
        }
        if value.dof_addresses.len() != local_basis_count {
            return Err(GalerkinAssemblyError::InvalidBasisDofs {
                element,
                dofs: value.dof_addresses.len(),
            });
        }
        for (local, &dof) in value.dof_addresses.iter().enumerate() {
            if dof >= global_dof_count {
                return Err(GalerkinAssemblyError::GlobalDofOutOfRange {
                    element,
                    local,
                    dof,
                    global_dof_count,
                });
            }
            if value.dof_addresses[..local].contains(&dof) {
                return Err(GalerkinAssemblyError::DuplicateGlobalDof {
                    element,
                    local,
                    dof,
                });
            }
        }
        active.push(element);
    }
    if active.is_empty() {
        return Err(GalerkinAssemblyError::EmptySurface);
    }
    Ok(active)
}

#[allow(dead_code)]
fn validate_closed_quad4_hypersingular_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    local_basis_count: usize,
    global_dof_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    validate_quad4_hypersingular_layout(elements, nodes, local_basis_count, global_dof_count, true)
}

fn validate_open_quad4_hypersingular_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    local_basis_count: usize,
    global_dof_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    validate_quad4_hypersingular_layout(elements, nodes, local_basis_count, global_dof_count, false)
}

fn validate_quad4_hypersingular_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    local_basis_count: usize,
    global_dof_count: usize,
    require_closed: bool,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let active =
        active_quadrilateral_global_layout(elements, nodes, local_basis_count, global_dof_count)?;
    let mut edge_incidence = HashMap::<(usize, usize), usize>::new();
    let mut vertex_dofs = HashMap::<usize, usize>::new();
    // Keep the reverse maps as well as the entity-to-DOF maps.  Checking only
    // that a shared vertex/edge repeats the same address is insufficient: a
    // vertex and an unrelated edge could still alias one global DOF, silently
    // changing the trace space and corrupting the weak hypersingular form.
    let mut dof_vertices = HashMap::<usize, usize>::new();
    let mut edge_dofs = HashMap::<(usize, usize), usize>::new();
    let mut dof_edges = HashMap::<usize, (usize, usize)>::new();
    let mut dof_centers = HashMap::<usize, usize>::new();
    for &element_index in &active {
        let element = &elements[element_index];
        if element.connectivity.len() != 4 {
            return Err(GalerkinAssemblyError::InvalidHypersingularConnectivity {
                element: element_index,
                nodes: element.connectivity.len(),
            });
        }
        for local in 0..4 {
            let node = element.connectivity[local];
            let dof = element.dof_addresses[local];
            if let Some(&expected) = vertex_dofs.get(&node) {
                if expected != dof {
                    return Err(GalerkinAssemblyError::NonConformingHypersingularDof {
                        element: element_index,
                        local,
                        node,
                        actual: dof,
                        expected,
                    });
                }
            } else {
                vertex_dofs.insert(node, dof);
            }
            if dof_edges.contains_key(&dof) || dof_centers.contains_key(&dof) {
                return Err(
                    GalerkinAssemblyError::HypersingularGlobalDofEntityConflict {
                        element: element_index,
                        local,
                        dof,
                    },
                );
            }
            if let Some(&expected_node) = dof_vertices.get(&dof) {
                if expected_node != node {
                    return Err(GalerkinAssemblyError::NonConformingHypersingularDof {
                        element: element_index,
                        local,
                        node,
                        actual: dof,
                        expected: vertex_dofs.get(&expected_node).copied().unwrap_or(dof),
                    });
                }
            } else {
                dof_vertices.insert(dof, node);
            }
            if local_basis_count == 9 {
                let first = node;
                let second = element.connectivity[(local + 1) % 4];
                let key = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                let edge_dof = element.dof_addresses[4 + local];
                if let Some(&expected) = edge_dofs.get(&key) {
                    if expected != edge_dof {
                        return Err(GalerkinAssemblyError::NonConformingHypersingularEdgeDof {
                            element: element_index,
                            first: key.0,
                            second: key.1,
                            actual: edge_dof,
                            expected,
                        });
                    }
                } else {
                    edge_dofs.insert(key, edge_dof);
                }
                if dof_vertices.contains_key(&edge_dof)
                    || dof_centers.contains_key(&edge_dof)
                    || dof_edges
                        .get(&edge_dof)
                        .is_some_and(|expected| *expected != key)
                {
                    return Err(
                        GalerkinAssemblyError::HypersingularGlobalDofEntityConflict {
                            element: element_index,
                            local: 4 + local,
                            dof: edge_dof,
                        },
                    );
                }
                dof_edges.insert(edge_dof, key);
            }
        }
        if local_basis_count == 9 {
            let center_dof = element.dof_addresses[8];
            if dof_vertices.contains_key(&center_dof)
                || dof_edges.contains_key(&center_dof)
                || dof_centers.contains_key(&center_dof)
            {
                return Err(
                    GalerkinAssemblyError::HypersingularGlobalDofEntityConflict {
                        element: element_index,
                        local: 8,
                        dof: center_dof,
                    },
                );
            }
            dof_centers.insert(center_dof, element_index);
        }
        for edge in 0..4 {
            let first = element.connectivity[edge];
            let second = element.connectivity[(edge + 1) % 4];
            let key = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            *edge_incidence.entry(key).or_default() += 1;
        }
    }
    if require_closed
        && let Some((&(first, second), &incidence)) = edge_incidence
            .iter()
            .find(|&(_, &incidence)| incidence != 2)
    {
        return Err(GalerkinAssemblyError::OpenHypersingularSurface {
            first,
            second,
            incidence,
        });
    }
    validate_quad4_hypersingular_orientation(elements, &active)?;
    Ok(active)
}

fn active_triangular_global_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    local_basis_count: usize,
    global_dof_count: usize,
) -> Result<Vec<usize>, GalerkinAssemblyError> {
    let mut active = Vec::new();
    for (element, value) in elements.iter().enumerate() {
        if value.property.is_evaluation() {
            continue;
        }
        if value.element_type != ElementType::Tri3 {
            return Err(GalerkinAssemblyError::UnsupportedElementType {
                element,
                element_type: value.element_type,
            });
        }
        if value.connectivity.len() != 3 {
            return Err(GalerkinAssemblyError::InvalidTri3Connectivity {
                element,
                nodes: value.connectivity.len(),
            });
        }
        for (local, &node) in value.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(GalerkinAssemblyError::NodeOutOfRange {
                    element,
                    local,
                    node,
                    node_count: nodes.len(),
                });
            }
        }
        if value.dof_addresses.len() != local_basis_count {
            return Err(if local_basis_count == 1 {
                GalerkinAssemblyError::NonConstantPanel {
                    element,
                    dofs: value.dof_addresses.len(),
                }
            } else {
                GalerkinAssemblyError::InvalidBasisDofs {
                    element,
                    dofs: value.dof_addresses.len(),
                }
            });
        }
        for (local, &dof) in value.dof_addresses.iter().enumerate() {
            if dof >= global_dof_count {
                return Err(GalerkinAssemblyError::GlobalDofOutOfRange {
                    element,
                    local,
                    dof,
                    global_dof_count,
                });
            }
            if value.dof_addresses[..local].contains(&dof) {
                return Err(GalerkinAssemblyError::DuplicateGlobalDof {
                    element,
                    local,
                    dof,
                });
            }
        }
        active.push(element);
    }
    if active.is_empty() {
        return Err(GalerkinAssemblyError::EmptySurface);
    }
    Ok(active)
}
