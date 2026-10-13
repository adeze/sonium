//! Pure boundary integral operators V (single-layer) and W (hyper-singular).
//!
//! These complement `tbem`, which assembles the *combined* Burton-Miller
//! operator. The plain V and W matrices here are needed by the Calderón
//! opposite-order preconditioner (`bem-step1.md` §1.2): for a system involving
//! V the preconditioner is `M⁻¹ W M⁻¹`, and for one involving W it is
//! `M⁻¹ V M⁻¹`.
//!
//! Convention:
//! - `V[i, j] = ∫_{T_j} G(x_i, y) dS_y`
//! - `W[i, j] = ∫_{T_j} ∂²G(x_i, y) / (∂n_x ∂n_y) dS_y`
//!
//! Unlike the single- and double-layer traces, the hypersingular boundary
//! operator has no `±1/2 I` jump term. Adding one breaks `W[1] = 0` in the
//! Laplace/static limit and corrupts opposite-order Calderón preconditioning.
//!
//! Singular self-element integrals are evaluated through `singular_integration`
//! (Sauter–Schwab / Duffy regularisation), exactly as in `tbem`. Off-diagonal
//! entries use `regular_integration`.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::core::assembly::common::get_element_coords;
use crate::core::integration::{
    regular_integration, regular_integration_linear_wavenumber_derivative_with_normals,
    regular_integration_linear_with_normals, regular_integration_wavenumber_derivative,
    singular_integration, singular_integration_linear_curved,
    singular_integration_linear_curved_wavenumber_derivative,
    singular_integration_wavenumber_derivative,
};
use crate::core::parallel::{
    BEM_QUADRATURE_PAIRS_PER_THREAD, gated_enumerate_filter_map, gated_map, use_parallel,
};
use crate::core::types::{BasisOrder, Element, IntegrationResult, PhysicsParams};
use sonium_solvers::dense::DenseMatrix;

/// The four constant-panel boundary integral operators evaluated with one
/// shared quadrature pass.
///
/// The kernels use the outward normal stored on the collocation/source panel
/// and on the field/integration panel.  `double_layer` is the principal-value
/// `K` operator (`∂G/∂n_y`), while `adjoint_double_layer` is `K'`
/// (`∂G/∂n_x`).  The free `±1/2 I` trace terms are deliberately not included;
/// callers add the trace convention appropriate to their formulation.
#[derive(Debug, Clone)]
pub struct BoundaryOperators {
    /// Single-layer operator `V`.
    pub single_layer: DenseMatrix<Complex64>,
    /// Double-layer operator `K`.
    pub double_layer: DenseMatrix<Complex64>,
    /// Adjoint double-layer operator `K'`.
    pub adjoint_double_layer: DenseMatrix<Complex64>,
    /// Hypersingular operator `W`.
    pub hypersingular: DenseMatrix<Complex64>,
}

fn count_active_dofs(elements: &[Element]) -> usize {
    elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .map(|e| e.dof_addresses.len())
        .sum()
}

/// Hard guard against multi-DOF (linear / higher-order) basis: the V/W
/// assemblers below collocate at the element centre and therefore only emit
/// one row/column per element. With a `BasisOrder::Linear` mesh this would
/// silently leave 2 of the 3 nodal DOFs unfilled. Crash early instead.
fn assert_single_dof_per_element(elements: &[Element]) {
    for (idx, e) in elements.iter().enumerate() {
        if e.property.is_evaluation() {
            continue;
        }
        let n = e.dof_addresses.len();
        assert!(
            n == 1,
            "operators.rs (V/W assembler): element {idx} carries {n} DOFs; \
             only constant-element (1 DOF/panel) basis is supported. \
             For linear / higher-order basis a per-basis collocation/Galerkin \
             implementation is required."
        );
    }
}

/// Validate the layout consumed by the legacy infallible constant-panel
/// assembler before a checked API delegates to it.
fn validate_constant_boundary_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
) -> Result<(), BoundaryOperatorAssemblyError> {
    let mut active_count = 0usize;
    let mut dof_owners = Vec::new();

    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }

        let expected_dofs = element.num_dofs_for_basis(BasisOrder::Constant);
        if element.dof_addresses.len() != expected_dofs {
            return Err(BoundaryOperatorAssemblyError::BasisDofCount {
                element: element_index,
                actual: element.dof_addresses.len(),
                expected: expected_dofs,
                basis: BasisOrder::Constant,
            });
        }

        let expected_nodes = element.element_type.num_nodes();
        if element.connectivity.len() != expected_nodes {
            return Err(BoundaryOperatorAssemblyError::ConnectivityNodeCount {
                element: element_index,
                actual: element.connectivity.len(),
                expected: expected_nodes,
            });
        }
        for (local, &node) in element.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(BoundaryOperatorAssemblyError::NodeOutOfRange {
                    element: element_index,
                    local,
                    node,
                    node_count: nodes.len(),
                });
            }
        }

        let dof = element.dof_addresses[0];
        if let Some((first_element, _)) = dof_owners.iter().find(|(_, owner_dof)| *owner_dof == dof)
        {
            return Err(BoundaryOperatorAssemblyError::DuplicateDof {
                first_element: *first_element,
                second_element: element_index,
                dof,
            });
        }
        dof_owners.push((element_index, dof));
        active_count += 1;
    }

    if active_count == 0 {
        return Err(BoundaryOperatorAssemblyError::EmptySurface);
    }

    for expected in 0..active_count {
        if !dof_owners.iter().any(|(_, dof)| *dof == expected) {
            let (element, actual) = dof_owners
                .iter()
                .find(|(_, dof)| *dof > expected)
                .copied()
                .unwrap_or(dof_owners[active_count - 1]);
            return Err(BoundaryOperatorAssemblyError::DofLayout {
                element,
                actual,
                expected,
            });
        }
    }

    Ok(())
}

fn precompute_coords(elements: &[Element], nodes: &[[f64; 3]]) -> Vec<DenseMatrix<f64>> {
    elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect()
}

/// Assemble all four constant-panel boundary operators in one production
/// quadrature pass.
///
/// Evaluation panels are skipped.  As with the historical `V`/`W` builders,
/// every active element must carry exactly one DOF and active DOF addresses
/// must index the returned dense matrices.
pub fn assemble_boundary_operators(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> BoundaryOperators {
    assert_single_dof_per_element(elements);
    let n = count_active_dofs(elements);
    let coords = precompute_coords(elements, nodes);

    // Row assembly (rayon above the machine-adaptive gate, serial for
    // small systems). Work is source × field kernel evaluations.
    let rows: Vec<(usize, Vec<(usize, IntegrationResult)>)> = gated_enumerate_filter_map(
        elements,
        n.saturating_mul(n),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |source_index, source| {
            if source.property.is_evaluation() {
                return None;
            }
            let source_dof = source.dof_addresses[0];
            let mut row_entries = Vec::with_capacity(n);

            for (field_index, field) in elements.iter().enumerate() {
                if field.property.is_evaluation() {
                    continue;
                }
                let result = if source_index == field_index {
                    singular_integration(
                        &source.center,
                        &source.normal,
                        &coords[field_index],
                        field.element_type,
                        physics,
                        None,
                        0,
                        false,
                    )
                } else {
                    regular_integration(
                        &source.center,
                        &source.normal,
                        &coords[field_index],
                        field.element_type,
                        field.area,
                        physics,
                        None,
                        0,
                        false,
                    )
                };
                row_entries.push((field.dof_addresses[0], result));
            }
            Some((source_dof, row_entries))
        },
    );

    let mut operators = BoundaryOperators {
        single_layer: DenseMatrix::zeros(n, n),
        double_layer: DenseMatrix::zeros(n, n),
        adjoint_double_layer: DenseMatrix::zeros(n, n),
        hypersingular: DenseMatrix::zeros(n, n),
    };
    for (source_dof, entries) in rows {
        for (field_dof, result) in entries {
            operators.single_layer[[source_dof, field_dof]] = result.g_integral;
            operators.double_layer[[source_dof, field_dof]] = result.dg_dn_integral;
            operators.adjoint_double_layer[[source_dof, field_dof]] = result.dg_dnx_integral;
            operators.hypersingular[[source_dof, field_dof]] = result.d2g_dnxdny_integral;
        }
    }
    operators
}

/// Assemble the analytic derivative of all four constant-panel boundary
/// operators with respect to the medium wave number.
///
/// Geometry, panel topology, and quadrature schedules are held fixed.  The
/// derivative is taken through the acoustic kernels; medium prefactors that
/// callers apply to the operator blocks are intentionally left to the caller.
pub fn assemble_boundary_operators_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> Result<BoundaryOperators, BoundaryOperatorAssemblyError> {
    validate_constant_boundary_layout(elements, nodes)?;

    let active: Vec<(usize, usize)> = elements
        .iter()
        .enumerate()
        .filter(|(_, element)| !element.property.is_evaluation())
        .map(|(index, element)| (index, element.dof_addresses[0]))
        .collect();
    let n = active.len();
    let coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();

    let rows: Vec<(usize, Vec<(usize, IntegrationResult)>)> = gated_map(
        &active,
        n.saturating_mul(n),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |&(source_index, source_dof)| {
            let source = &elements[source_index];
            let row_entries = active
                .iter()
                .map(|&(field_index, field_dof)| {
                    let field = &elements[field_index];
                    let result = if source_index == field_index {
                        singular_integration_wavenumber_derivative(
                            &source.center,
                            &source.normal,
                            &coords[field_index],
                            field.element_type,
                            physics,
                            None,
                            0,
                            false,
                            None,
                        )
                    } else {
                        regular_integration_wavenumber_derivative(
                            &source.center,
                            &source.normal,
                            &coords[field_index],
                            field.element_type,
                            field.area,
                            physics,
                            None,
                            0,
                            false,
                            None,
                        )
                    };
                    (field_dof, result)
                })
                .collect();
            (source_dof, row_entries)
        },
    );

    let mut operators = BoundaryOperators {
        single_layer: DenseMatrix::zeros(n, n),
        double_layer: DenseMatrix::zeros(n, n),
        adjoint_double_layer: DenseMatrix::zeros(n, n),
        hypersingular: DenseMatrix::zeros(n, n),
    };
    for (source_dof, entries) in rows {
        for (field_dof, result) in entries {
            operators.single_layer[[source_dof, field_dof]] = result.g_integral;
            operators.double_layer[[source_dof, field_dof]] = result.dg_dn_integral;
            operators.adjoint_double_layer[[source_dof, field_dof]] = result.dg_dnx_integral;
            operators.hypersingular[[source_dof, field_dof]] = result.d2g_dnxdny_integral;
        }
    }
    Ok(operators)
}

/// Errors returned by the basis-aware boundary-operator assembler.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryOperatorAssemblyError {
    /// An active element does not expose the number of local basis functions
    /// required by the selected basis order.
    #[error("element {element} has {actual} DOFs, expected {expected} for {basis:?}")]
    BasisDofCount {
        /// Active element index.
        element: usize,
        /// Number of addresses supplied by the caller.
        actual: usize,
        /// Number required by the selected basis.
        expected: usize,
        /// Requested panel basis.
        basis: BasisOrder,
    },
    /// An element references a mesh node that is not present in `nodes`.
    #[error(
        "element {element} local node {local} references node {node}, but mesh has {node_count} nodes"
    )]
    NodeOutOfRange {
        /// Element containing the invalid connectivity entry.
        element: usize,
        /// Local connectivity slot containing the invalid node address.
        local: usize,
        /// Invalid global node address.
        node: usize,
        /// Number of available mesh nodes.
        node_count: usize,
    },
    /// An element's connectivity length does not match its element type.
    #[error(
        "element {element} has {actual} connectivity nodes; {expected} required for its element type"
    )]
    ConnectivityNodeCount {
        /// Element with malformed connectivity.
        element: usize,
        /// Number of supplied connectivity entries.
        actual: usize,
        /// Number required by the element type.
        expected: usize,
    },
    /// Two active constant panels claim the same global DOF address.
    #[error("constant-panel DOF {dof} is used by elements {first_element} and {second_element}")]
    DuplicateDof {
        /// First element that claimed the address.
        first_element: usize,
        /// Later element that reused the address.
        second_element: usize,
        /// Reused global DOF address.
        dof: usize,
    },
    /// Active DOF addresses must form a dense zero-based layout.
    #[error("active DOF layout has a gap at index {expected}; element {element} supplied {actual}")]
    DofLayout {
        /// Element containing the first unexpected address.
        element: usize,
        /// Address encountered while checking the layout.
        actual: usize,
        /// Address required by dense ordering.
        expected: usize,
    },
    /// No active panels were supplied.
    #[error("boundary-operator assembly requires at least one active element")]
    EmptySurface,
}

/// Assemble all four boundary operators with discontinuous or globally dense
/// linear/quadratic panel coefficients.
///
/// Rows are collocated at the selected panel basis points and columns retain
/// the basis-weighted regular/singular kernel integrals. Existing DOF
/// addresses are honored, so a continuous quadratic map may share rows and
/// columns while a discontinuous layout remains panel-local. The constant
/// panel API remains available through [`assemble_boundary_operators`].
pub fn assemble_boundary_operators_with_basis(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
) -> Result<BoundaryOperators, BoundaryOperatorAssemblyError> {
    if basis == BasisOrder::Constant {
        validate_constant_boundary_layout(elements, nodes)?;
        return Ok(assemble_boundary_operators(elements, nodes, physics));
    }

    let mut active_count = 0usize;
    let mut max_dof: Option<usize> = None;
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let expected = element.num_dofs_for_basis(basis);
        if element.dof_addresses.len() != expected {
            return Err(BoundaryOperatorAssemblyError::BasisDofCount {
                element: element_index,
                actual: element.dof_addresses.len(),
                expected,
                basis,
            });
        }
        active_count += 1;
        for &dof in &element.dof_addresses {
            max_dof = Some(max_dof.map_or(dof, |max_value| max_value.max(dof)));
        }
    }
    if active_count == 0 {
        return Err(BoundaryOperatorAssemblyError::EmptySurface);
    }
    let n = max_dof.expect("active_count guarantees a DOF") + 1;
    let mut seen = vec![None; n];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        for &dof in &element.dof_addresses {
            if let Some(previous) = seen[dof] {
                if previous == element_index {
                    continue;
                }
            } else {
                seen[dof] = Some(element_index);
            }
        }
    }
    if let Some(expected) = seen.iter().position(Option::is_none) {
        return Err(BoundaryOperatorAssemblyError::DofLayout {
            element: elements
                .iter()
                .position(|element| !element.property.is_evaluation())
                .unwrap_or(0),
            actual: n,
            expected,
        });
    }

    let coords = precompute_coords(elements, nodes);
    // Per-source row batches (rayon above the machine-adaptive gate, serial
    // for small systems). The closure is shared by both paths so the row
    // batches are identical.
    let build_rows = |(source_index, source): (usize, &Element)| {
        if source.property.is_evaluation() {
            return Vec::new();
        }
        let coords = &coords;
        let source_points = source.collocation_points(basis, nodes);
        let source_normals = vec![source.normal; source_points.len()];
        source_points
            .into_iter()
            .zip(source_normals)
            .zip(source.dof_addresses.iter().copied())
            .map(move |((source_point, source_normal), source_dof)| {
                let mut row_entries = Vec::new();
                for (field_index, field) in elements.iter().enumerate() {
                    if field.property.is_evaluation() {
                        continue;
                    }
                    let num_field_dofs = field.num_dofs_for_basis(basis);
                    let linear_result = if source_index == field_index {
                        singular_integration_linear_curved(
                            &source_point,
                            &source_normal,
                            &coords[field_index],
                            field.element_type,
                            physics,
                            num_field_dofs,
                            None,
                        )
                    } else {
                        regular_integration_linear_with_normals(
                            &source_point,
                            &source_normal,
                            &coords[field_index],
                            field.element_type,
                            field.area,
                            physics,
                            num_field_dofs,
                            None,
                        )
                    };
                    row_entries.extend(
                        field
                            .dof_addresses
                            .iter()
                            .copied()
                            .zip(linear_result.basis_results),
                    );
                }
                (source_dof, row_entries)
            })
            .collect::<Vec<_>>()
    };
    let rows: Vec<(usize, Vec<(usize, IntegrationResult)>)> =
        if use_parallel(n.saturating_mul(elements.len()), BEM_QUADRATURE_PAIRS_PER_THREAD) {
            elements
                .par_iter()
                .enumerate()
                .flat_map_iter(&build_rows)
                .collect()
        } else {
            elements.iter().enumerate().flat_map(&build_rows).collect()
        };

    let mut operators = BoundaryOperators {
        single_layer: DenseMatrix::zeros(n, n),
        double_layer: DenseMatrix::zeros(n, n),
        adjoint_double_layer: DenseMatrix::zeros(n, n),
        hypersingular: DenseMatrix::zeros(n, n),
    };
    for (source_dof, entries) in rows {
        for (field_dof, result) in entries {
            operators.single_layer[[source_dof, field_dof]] += result.g_integral;
            operators.double_layer[[source_dof, field_dof]] += result.dg_dn_integral;
            operators.adjoint_double_layer[[source_dof, field_dof]] += result.dg_dnx_integral;
            operators.hypersingular[[source_dof, field_dof]] += result.d2g_dnxdny_integral;
        }
    }
    Ok(operators)
}

/// Assemble the dense single-layer operator V on the active surface DOFs.
///
/// **Constant-element basis only.** Evaluation panels
/// (`property.is_evaluation() == true`) are skipped. Indexed by each
/// element's single `dof_addresses[0]`. Panics on linear / higher-order
/// elements — see `assert_single_dof_per_element`.
pub fn assemble_single_layer(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> DenseMatrix<Complex64> {
    assert_single_dof_per_element(elements);
    let n = count_active_dofs(elements);
    let coords = precompute_coords(elements, nodes);

    // Build per-row vectors (rayon above the machine-adaptive gate, serial
    // for small systems); the kernel evaluations are independent per
    // source element.
    let rows: Vec<(usize, Vec<(usize, Complex64)>)> = gated_enumerate_filter_map(
        elements,
        n.saturating_mul(n),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |iel, src| {
            if src.property.is_evaluation() {
                return None;
            }
            let source_dof = src.dof_addresses[0];
            let source_point = &src.center;
            let source_normal = &src.normal;
            let mut row_entries: Vec<(usize, Complex64)> = Vec::with_capacity(n);

            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                let field_dof = fld.dof_addresses[0];
                let result = if iel == jel {
                    singular_integration(
                        source_point,
                        source_normal,
                        &coords[jel],
                        fld.element_type,
                        physics,
                        None,
                        0,
                        false,
                    )
                } else {
                    regular_integration(
                        source_point,
                        source_normal,
                        &coords[jel],
                        fld.element_type,
                        fld.area,
                        physics,
                        None,
                        0,
                        false,
                    )
                };
                row_entries.push((field_dof, result.g_integral));
            }
            Some((source_dof, row_entries))
        },
    );

    let mut v = DenseMatrix::<Complex64>::zeros(n, n);
    for (i, entries) in rows {
        for (j, val) in entries {
            v[[i, j]] = val;
        }
    }
    v
}

/// Assemble the dense hyper-singular operator W on the active surface DOFs.
///
/// **Constant-element basis only** — see `assert_single_dof_per_element`.
/// The hyper-singular kernel is evaluated through the same
/// `regular_integration` / `singular_integration` path as TBEM.
pub fn assemble_hypersingular(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> DenseMatrix<Complex64> {
    assert_single_dof_per_element(elements);
    let n = count_active_dofs(elements);
    let coords = precompute_coords(elements, nodes);

    let rows: Vec<(usize, Vec<(usize, Complex64)>)> = gated_enumerate_filter_map(
        elements,
        n.saturating_mul(n),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |iel, src| {
            if src.property.is_evaluation() {
                return None;
            }
            let source_dof = src.dof_addresses[0];
            let source_point = &src.center;
            let source_normal = &src.normal;
            let mut row_entries: Vec<(usize, Complex64)> = Vec::with_capacity(n);

            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                let field_dof = fld.dof_addresses[0];
                let result = if iel == jel {
                    singular_integration(
                        source_point,
                        source_normal,
                        &coords[jel],
                        fld.element_type,
                        physics,
                        None,
                        1,
                        false,
                    )
                } else {
                    regular_integration(
                        source_point,
                        source_normal,
                        &coords[jel],
                        fld.element_type,
                        fld.area,
                        physics,
                        None,
                        1,
                        false,
                    )
                };
                row_entries.push((field_dof, result.d2g_dnxdny_integral));
            }
            Some((source_dof, row_entries))
        },
    );

    let mut w = DenseMatrix::<Complex64>::zeros(n, n);
    for (i, entries) in rows {
        for (j, val) in entries {
            w[[i, j]] = val;
        }
    }
    w
}

pub(crate) fn active_node_normals(element: &Element) -> Option<&[[f64; 3]]> {
    (element.node_normals.len() == element.num_nodes()
        && element
            .node_normals
            .iter()
            .any(|normal| normal.iter().any(|value| value.abs() > 1.0e-14)))
    .then_some(element.node_normals.as_slice())
}

/// Assemble basis-weighted boundary operators for a dense shared-DOF layout.
///
/// Each global DOF contributes one deterministic collocation row (the first
/// panel/local basis slot that owns it), while every panel occurrence is
/// accumulated into the field columns. This is the collocation analogue of
/// the conforming quadratic TBEM path: shared vertex/edge DOFs are not
/// overwritten when the same column appears on adjacent panels.
pub fn assemble_boundary_operators_with_basis_shared_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
) -> Result<BoundaryOperators, BoundaryOperatorAssemblyError> {
    if basis == BasisOrder::Constant {
        validate_constant_boundary_layout(elements, nodes)?;
        return Ok(assemble_boundary_operators(elements, nodes, physics));
    }

    let mut active_count = 0usize;
    let mut max_dof = None;
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let expected = element.num_dofs_for_basis(basis);
        if element.dof_addresses.len() != expected {
            return Err(BoundaryOperatorAssemblyError::BasisDofCount {
                element: element_index,
                actual: element.dof_addresses.len(),
                expected,
                basis,
            });
        }
        active_count += 1;
        for &dof in &element.dof_addresses {
            max_dof = Some(max_dof.map_or(dof, |current: usize| current.max(dof)));
        }
    }
    if active_count == 0 {
        return Err(BoundaryOperatorAssemblyError::EmptySurface);
    }

    let n = max_dof.expect("active_count guarantees a DOF") + 1;
    let mut seen = vec![None; n];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        for &dof in &element.dof_addresses {
            if seen[dof].is_none() {
                seen[dof] = Some(element_index);
            }
        }
    }
    if let Some(expected) = seen.iter().position(Option::is_none) {
        return Err(BoundaryOperatorAssemblyError::DofLayout {
            element: elements
                .iter()
                .position(|element| !element.property.is_evaluation())
                .unwrap_or(0),
            actual: n,
            expected,
        });
    }

    let coords = precompute_coords(elements, nodes);
    let mut owners = vec![None; n];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            owners[dof].get_or_insert((element_index, local));
        }
    }

    let mut operators = BoundaryOperators {
        single_layer: DenseMatrix::zeros(n, n),
        double_layer: DenseMatrix::zeros(n, n),
        adjoint_double_layer: DenseMatrix::zeros(n, n),
        hypersingular: DenseMatrix::zeros(n, n),
    };

    for (source_dof, owner) in owners.into_iter().enumerate() {
        let Some((source_index, source_local)) = owner else {
            continue;
        };
        let source = &elements[source_index];
        let source_points = source.collocation_points(basis, nodes);
        let source_normals = source.collocation_normals(basis, nodes);
        let source_point = source_points
            .get(source_local)
            .expect("basis collocation points match local DOF count");
        let source_normal = source_normals
            .get(source_local)
            .copied()
            .unwrap_or(source.normal);

        for (field_index, field) in elements.iter().enumerate() {
            if field.property.is_evaluation() {
                continue;
            }
            let field_dofs = field.num_dofs_for_basis(basis);
            let field_normals = active_node_normals(field);
            let result = if source_index == field_index {
                singular_integration_linear_curved(
                    source_point,
                    &source_normal,
                    &coords[field_index],
                    field.element_type,
                    physics,
                    field_dofs,
                    field_normals,
                )
            } else {
                regular_integration_linear_with_normals(
                    source_point,
                    &source_normal,
                    &coords[field_index],
                    field.element_type,
                    field.area,
                    physics,
                    field_dofs,
                    field_normals,
                )
            };

            for (local_basis, field_result) in result.basis_results.into_iter().enumerate() {
                let field_dof = field.dof_addresses[local_basis];
                operators.single_layer[[source_dof, field_dof]] += field_result.g_integral;
                operators.double_layer[[source_dof, field_dof]] += field_result.dg_dn_integral;
                operators.adjoint_double_layer[[source_dof, field_dof]] +=
                    field_result.dg_dnx_integral;
                operators.hypersingular[[source_dof, field_dof]] +=
                    field_result.d2g_dnxdny_integral;
            }
        }
    }

    Ok(operators)
}

/// Assemble the analytic wavenumber derivative for a shared higher-order
/// basis layout.
///
/// This preserves the shared-DOF collocation contract of
/// [`assemble_boundary_operators_with_basis_shared_dofs`]: the first
/// panel/local occurrence owns each global row, while every panel occurrence
/// contributes to its field column.  Curved nodal normals and the adaptive
/// regular/singular quadrature schedules are held fixed while differentiating
/// the four acoustic kernels.
pub fn assemble_boundary_operators_wavenumber_derivative_with_basis_shared_dofs(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
) -> Result<BoundaryOperators, BoundaryOperatorAssemblyError> {
    if basis == BasisOrder::Constant {
        return assemble_boundary_operators_wavenumber_derivative(elements, nodes, physics);
    }

    let mut active_count = 0usize;
    let mut max_dof = None;
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let expected = element.num_dofs_for_basis(basis);
        if element.dof_addresses.len() != expected {
            return Err(BoundaryOperatorAssemblyError::BasisDofCount {
                element: element_index,
                actual: element.dof_addresses.len(),
                expected,
                basis,
            });
        }
        active_count += 1;
        for &dof in &element.dof_addresses {
            max_dof = Some(max_dof.map_or(dof, |current: usize| current.max(dof)));
        }
    }
    if active_count == 0 {
        return Err(BoundaryOperatorAssemblyError::EmptySurface);
    }

    let n = max_dof.expect("active_count guarantees a DOF") + 1;
    let mut seen = vec![None; n];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        for &dof in &element.dof_addresses {
            if let Some(previous) = seen[dof] {
                if previous == element_index {
                    continue;
                }
            } else {
                seen[dof] = Some(element_index);
            }
        }
    }
    if let Some(expected) = seen.iter().position(Option::is_none) {
        return Err(BoundaryOperatorAssemblyError::DofLayout {
            element: elements
                .iter()
                .position(|element| !element.property.is_evaluation())
                .unwrap_or(0),
            actual: n,
            expected,
        });
    }

    let coords = precompute_coords(elements, nodes);
    let mut owners = vec![None; n];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            owners[dof].get_or_insert((element_index, local));
        }
    }

    let mut operators = BoundaryOperators {
        single_layer: DenseMatrix::zeros(n, n),
        double_layer: DenseMatrix::zeros(n, n),
        adjoint_double_layer: DenseMatrix::zeros(n, n),
        hypersingular: DenseMatrix::zeros(n, n),
    };
    for (source_dof, owner) in owners.into_iter().enumerate() {
        let Some((source_index, source_local)) = owner else {
            continue;
        };
        let source = &elements[source_index];
        let source_point = source
            .collocation_points(basis, nodes)
            .get(source_local)
            .copied()
            .expect("basis collocation points match local DOF count");
        let source_normal = source
            .collocation_normals(basis, nodes)
            .get(source_local)
            .copied()
            .unwrap_or(source.normal);

        for (field_index, field) in elements.iter().enumerate() {
            if field.property.is_evaluation() {
                continue;
            }
            let field_dofs = field.num_dofs_for_basis(basis);
            let field_normals = active_node_normals(field);
            let derivative = if source_index == field_index {
                singular_integration_linear_curved_wavenumber_derivative(
                    &source_point,
                    &source_normal,
                    &coords[field_index],
                    field.element_type,
                    physics,
                    field_dofs,
                    field_normals,
                )
            } else {
                regular_integration_linear_wavenumber_derivative_with_normals(
                    &source_point,
                    &source_normal,
                    &coords[field_index],
                    field.element_type,
                    field.area,
                    physics,
                    field_dofs,
                    field_normals,
                )
            };
            for (local_basis, result) in derivative.basis_results.into_iter().enumerate() {
                let field_dof = field.dof_addresses[local_basis];
                operators.single_layer[[source_dof, field_dof]] += result.g_integral;
                operators.double_layer[[source_dof, field_dof]] += result.dg_dn_integral;
                operators.adjoint_double_layer[[source_dof, field_dof]] += result.dg_dnx_integral;
                operators.hypersingular[[source_dof, field_dof]] += result.d2g_dnxdny_integral;
            }
        }
    }
    Ok(operators)
}

/// Assemble the analytic wavenumber derivative for panel-local higher-order
/// basis operators.
///
/// The transmission builder uses panel-local DOFs for `BasisOrder::Linear`
/// and `BasisOrder::Quadratic`.  Rows follow the same per-panel collocation
/// order as [`assemble_boundary_operators_with_basis`], while every field
/// basis occurrence is retained in the corresponding columns.  Self panels
/// use the singular derivative quadrature; regular panels use the same
/// adaptive derivative schedule as the value assembler.
///
/// The transmission caller consumes this path for panel-local systems; shared
/// higher-order systems use the companion shared-DOF assembler above.
pub fn assemble_boundary_operators_wavenumber_derivative_with_basis(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
) -> Result<BoundaryOperators, BoundaryOperatorAssemblyError> {
    if basis == BasisOrder::Constant {
        return assemble_boundary_operators_wavenumber_derivative(elements, nodes, physics);
    }

    let mut active_count = 0usize;
    let mut max_dof = None;
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let expected = element.num_dofs_for_basis(basis);
        if element.dof_addresses.len() != expected {
            return Err(BoundaryOperatorAssemblyError::BasisDofCount {
                element: element_index,
                actual: element.dof_addresses.len(),
                expected,
                basis,
            });
        }
        active_count += 1;
        for &dof in &element.dof_addresses {
            max_dof = Some(max_dof.map_or(dof, |current: usize| current.max(dof)));
        }
    }
    if active_count == 0 {
        return Err(BoundaryOperatorAssemblyError::EmptySurface);
    }

    let n = max_dof.expect("active_count guarantees a DOF") + 1;
    let mut seen = vec![None; n];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        for &dof in &element.dof_addresses {
            if let Some(previous) = seen[dof] {
                if previous == element_index {
                    continue;
                }
            } else {
                seen[dof] = Some(element_index);
            }
        }
    }
    if let Some(expected) = seen.iter().position(Option::is_none) {
        return Err(BoundaryOperatorAssemblyError::DofLayout {
            element: elements
                .iter()
                .position(|element| !element.property.is_evaluation())
                .unwrap_or(0),
            actual: n,
            expected,
        });
    }

    let coords = precompute_coords(elements, nodes);
    // Per-source row batches (rayon above the machine-adaptive gate, serial
    // for small systems). The closure is shared by both paths so the row
    // batches are identical.
    let build_rows = |(source_index, source): (usize, &Element)| {
        if source.property.is_evaluation() {
            return Vec::new();
        }
        let coords = &coords;
        let source_points = source.collocation_points(basis, nodes);
        let source_normals = vec![source.normal; source_points.len()];
        source_points
            .into_iter()
            .zip(source_normals)
            .zip(source.dof_addresses.iter().copied())
            .map(move |((source_point, source_normal), source_dof)| {
                let mut row_entries = Vec::new();
                for (field_index, field) in elements.iter().enumerate() {
                    if field.property.is_evaluation() {
                        continue;
                    }
                    let num_field_dofs = field.num_dofs_for_basis(basis);
                    let derivative = if source_index == field_index {
                        singular_integration_linear_curved_wavenumber_derivative(
                            &source_point,
                            &source_normal,
                            &coords[field_index],
                            field.element_type,
                            physics,
                            num_field_dofs,
                            None,
                        )
                    } else {
                        regular_integration_linear_wavenumber_derivative_with_normals(
                            &source_point,
                            &source_normal,
                            &coords[field_index],
                            field.element_type,
                            field.area,
                            physics,
                            num_field_dofs,
                            None,
                        )
                    };
                    row_entries.extend(
                        field
                            .dof_addresses
                            .iter()
                            .copied()
                            .zip(derivative.basis_results),
                    );
                }
                (source_dof, row_entries)
            })
            .collect::<Vec<_>>()
    };
    let rows: Vec<(usize, Vec<(usize, IntegrationResult)>)> =
        if use_parallel(n.saturating_mul(elements.len()), BEM_QUADRATURE_PAIRS_PER_THREAD) {
            elements
                .par_iter()
                .enumerate()
                .flat_map_iter(&build_rows)
                .collect()
        } else {
            elements.iter().enumerate().flat_map(&build_rows).collect()
        };

    let mut operators = BoundaryOperators {
        single_layer: DenseMatrix::zeros(n, n),
        double_layer: DenseMatrix::zeros(n, n),
        adjoint_double_layer: DenseMatrix::zeros(n, n),
        hypersingular: DenseMatrix::zeros(n, n),
    };
    for (source_dof, entries) in rows {
        for (field_dof, result) in entries {
            operators.single_layer[[source_dof, field_dof]] += result.g_integral;
            operators.double_layer[[source_dof, field_dof]] += result.dg_dn_integral;
            operators.adjoint_double_layer[[source_dof, field_dof]] += result.dg_dnx_integral;
            operators.hypersingular[[source_dof, field_dof]] += result.d2g_dnxdny_integral;
        }
    }
    Ok(operators)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::generators::generate_icosphere_mesh;
    use crate::core::types::{BoundaryCondition, PhysicsParams};

    fn small_sphere_setup() -> (Vec<Element>, Vec<[f64; 3]>, PhysicsParams) {
        let mesh = generate_icosphere_mesh(0.2, 1);
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            e.dof_addresses = vec![i];
        }
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        (elements, mesh.nodes, physics)
    }

    #[test]
    fn single_layer_is_finite_and_nonzero() {
        let (elements, nodes, physics) = small_sphere_setup();
        let v = assemble_single_layer(&elements, &nodes, &physics);
        assert_eq!(v.nrows(), elements.len());
        assert_eq!(v.ncols(), elements.len());
        let nrm: f64 = v
            .as_slice()
            .iter()
            .map(|z| z.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(nrm.is_finite() && nrm > 0.0);
        for &z in v.as_slice().iter() {
            assert!(z.re.is_finite() && z.im.is_finite());
        }
    }

    #[test]
    fn hypersingular_is_finite() {
        let (elements, nodes, physics) = small_sphere_setup();
        let w = assemble_hypersingular(&elements, &nodes, &physics);
        assert_eq!(w.nrows(), elements.len());
        for i in 0..w.nrows() {
            assert!(w[[i, i]].re.is_finite());
            assert!(w[[i, i]].im.is_finite());
        }
    }

    #[test]
    fn hypersingular_annihilates_constant_in_static_limit() {
        let radius = 0.2;
        let mesh = generate_icosphere_mesh(radius, 1);
        let mut elements = mesh.elements.clone();
        for (i, element) in elements.iter_mut().enumerate() {
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            element.dof_addresses = vec![i];
        }
        let speed_of_sound = 343.0;
        let ka = 1e-4;
        let frequency = ka * speed_of_sound / (2.0 * std::f64::consts::PI * radius);
        let physics = PhysicsParams::new(frequency, speed_of_sound, 1.21, false);
        let w = assemble_hypersingular(&elements, &mesh.nodes, &physics);

        let mean_row_sum_sqr = (0..w.nrows())
            .map(|row| {
                (0..w.ncols())
                    .map(|column| w[[row, column]])
                    .sum::<Complex64>()
                    .norm_sqr()
            })
            .sum::<f64>()
            / w.nrows() as f64;
        let rms_row_sum = mean_row_sum_sqr.sqrt();
        assert!(
            rms_row_sum < 0.1,
            "static hypersingular W[1] must approach zero, got RMS row sum {rms_row_sum:.3e}"
        );
    }
}
