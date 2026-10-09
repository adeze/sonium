//! Shared helper functions for BEM assembly modules.
//!
//! This module deduplicates logic that was previously copied between
//! `tbem.rs`, `slfmm.rs`, `hmatrix.rs`, and `mlfmm.rs`.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;

use crate::core::integration::{
    regular_integration, regular_integration_linear, regular_integration_linear_with_normals,
    singular_integration, singular_integration_linear, singular_integration_linear_curved,
};
use crate::core::types::{BasisOrder, BoundaryCondition, Element, ElementType, PhysicsParams};

/// Action selected when checking a matrix-free FMM operator against an
/// independently assembled reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmmAction {
    /// Forward product `A*x`.
    Forward,
    /// Algebraic transpose product `Aᵀ*x`.
    Transpose,
    /// Hermitian transpose product `Aᴴ*x`.
    Hermitian,
}

/// Checked failures returned by the FMM independent-reference accuracy gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmmAccuracyError {
    /// At least one deterministic probe is required.
    EmptyProbeSet,
    /// The reference output count differs from the probe count.
    ProbeCountMismatch {
        /// Number of supplied probes.
        probes: usize,
        /// Number of supplied reference outputs.
        references: usize,
    },
    /// A probe vector has the wrong operator dimension.
    InputLength {
        /// Probe index.
        probe: usize,
        /// Required operator dimension.
        expected: usize,
        /// Supplied vector length.
        actual: usize,
    },
    /// A probe contains a non-finite value.
    NonFiniteInput {
        /// Probe index.
        probe: usize,
        /// Index of the non-finite value.
        index: usize,
    },
    /// A reference output has the wrong operator dimension.
    ReferenceLength {
        /// Probe/reference index.
        probe: usize,
        /// Required operator dimension.
        expected: usize,
        /// Supplied vector length.
        actual: usize,
    },
    /// A reference output contains a non-finite value.
    NonFiniteReference {
        /// Probe/reference index.
        probe: usize,
        /// Index of the non-finite value.
        index: usize,
    },
    /// The FMM action returned the wrong operator dimension.
    OutputLength {
        /// Probe index.
        probe: usize,
        /// Required operator dimension.
        expected: usize,
        /// Computed vector length.
        actual: usize,
    },
    /// The FMM action returned a non-finite value.
    NonFiniteOutput {
        /// Probe index.
        probe: usize,
        /// Index of the non-finite value.
        index: usize,
    },
    /// The relative-error calculation produced a non-finite value.
    NonFiniteError {
        /// Probe/reference index.
        probe: usize,
    },
}

impl std::fmt::Display for FmmAccuracyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyProbeSet => write!(f, "FMM accuracy check requires at least one probe"),
            Self::ProbeCountMismatch { probes, references } => write!(
                f,
                "FMM accuracy check received {probes} probes and {references} references"
            ),
            Self::InputLength {
                probe,
                expected,
                actual,
            } => write!(
                f,
                "FMM probe {probe} has length {actual}, expected {expected}"
            ),
            Self::NonFiniteInput { probe, index } => {
                write!(f, "FMM probe {probe} value at index {index} is non-finite")
            }
            Self::ReferenceLength {
                probe,
                expected,
                actual,
            } => write!(
                f,
                "FMM reference output {probe} has length {actual}, expected {expected}"
            ),
            Self::NonFiniteReference { probe, index } => write!(
                f,
                "FMM reference output {probe} value at index {index} is non-finite"
            ),
            Self::OutputLength {
                probe,
                expected,
                actual,
            } => write!(
                f,
                "FMM action output {probe} has length {actual}, expected {expected}"
            ),
            Self::NonFiniteOutput { probe, index } => write!(
                f,
                "FMM action output {probe} value at index {index} is non-finite"
            ),
            Self::NonFiniteError { probe } => {
                write!(f, "FMM relative error for probe {probe} is non-finite")
            }
        }
    }
}

impl std::error::Error for FmmAccuracyError {}

/// Validate probes/references and compute the maximum relative action error.
pub(crate) fn checked_fmm_relative_action_error<F>(
    action: FmmAction,
    probes: &[Vec<Complex64>],
    reference_outputs: &[Vec<Complex64>],
    expected_len: usize,
    mut apply: F,
) -> Result<f64, FmmAccuracyError>
where
    F: FnMut(FmmAction, &[Complex64]) -> Vec<Complex64>,
{
    if probes.is_empty() {
        return Err(FmmAccuracyError::EmptyProbeSet);
    }
    if probes.len() != reference_outputs.len() {
        return Err(FmmAccuracyError::ProbeCountMismatch {
            probes: probes.len(),
            references: reference_outputs.len(),
        });
    }

    let mut maximum_error = 0.0_f64;
    for (probe_index, (probe, reference)) in probes.iter().zip(reference_outputs).enumerate() {
        if probe.len() != expected_len {
            return Err(FmmAccuracyError::InputLength {
                probe: probe_index,
                expected: expected_len,
                actual: probe.len(),
            });
        }
        if let Some(index) = probe
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(FmmAccuracyError::NonFiniteInput {
                probe: probe_index,
                index,
            });
        }
        if reference.len() != expected_len {
            return Err(FmmAccuracyError::ReferenceLength {
                probe: probe_index,
                expected: expected_len,
                actual: reference.len(),
            });
        }
        if let Some(index) = reference
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(FmmAccuracyError::NonFiniteReference {
                probe: probe_index,
                index,
            });
        }

        let output = apply(action, probe);
        if output.len() != expected_len {
            return Err(FmmAccuracyError::OutputLength {
                probe: probe_index,
                expected: expected_len,
                actual: output.len(),
            });
        }
        if let Some(index) = output
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(FmmAccuracyError::NonFiniteOutput {
                probe: probe_index,
                index,
            });
        }

        let difference_squared = output
            .iter()
            .zip(reference)
            .map(|(computed, expected)| (*computed - *expected).norm_sqr())
            .sum::<f64>();
        let reference_squared = reference.iter().map(|value| value.norm_sqr()).sum::<f64>();
        let relative_error =
            difference_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE);
        if !relative_error.is_finite() {
            return Err(FmmAccuracyError::NonFiniteError { probe: probe_index });
        }
        maximum_error = maximum_error.max(relative_error);
    }

    Ok(maximum_error)
}

// Higher-order FMM/H-matrix proxy elements encode their panel basis size and
// local slot in `Element::group`. Keep this decoder in the shared assembly
// helpers so every accelerated backend interprets the proxy layout identically.
const DOF_PROXY_TAG: usize = 1usize << (usize::BITS - 1);
const DOF_PROXY_BASIS_SHIFT: usize = 8;
const DOF_PROXY_BASIS_MASK: usize = 0xffusize << DOF_PROXY_BASIS_SHIFT;
const DOF_PROXY_LOCAL_MASK: usize = 0xff;

pub(crate) fn decode_dof_proxy_group(group: usize) -> Option<(usize, usize)> {
    if group & DOF_PROXY_TAG == 0 {
        return None;
    }
    let num_basis = (group & DOF_PROXY_BASIS_MASK) >> DOF_PROXY_BASIS_SHIFT;
    let local = group & DOF_PROXY_LOCAL_MASK;
    (num_basis > 1).then_some((num_basis, local))
}

pub(crate) fn basis_order_for_proxy(element: &Element) -> Option<BasisOrder> {
    let num_basis = decode_dof_proxy_group(element.group)
        .map(|(count, _)| count)
        .unwrap_or_else(|| element.dof_addresses.len());
    match (element.element_type, num_basis) {
        (ElementType::Tri3, 3) | (ElementType::Quad4, 4) => Some(BasisOrder::Linear),
        (ElementType::Tri3, 6) | (ElementType::Quad4, 9) => Some(BasisOrder::Quadratic),
        (ElementType::Tri3 | ElementType::Quad4, 1) => Some(BasisOrder::Constant),
        _ => None,
    }
}

pub(crate) fn proxy_local_index(element: &Element) -> usize {
    decode_dof_proxy_group(element.group)
        .map(|(_, local)| local)
        .unwrap_or(0)
}

/// Count total number of DOFs (excluding evaluation elements).
pub fn count_dofs(elements: &[Element]) -> usize {
    elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .flat_map(|e| e.dof_addresses.iter().copied())
        .max()
        .map_or(0, |max_dof| max_dof + 1)
}

/// Get boundary condition type (0 = velocity, 1 = pressure) and values.
pub fn get_bc_type_and_value(bc: &BoundaryCondition) -> (i32, Vec<Complex64>) {
    match bc {
        BoundaryCondition::Velocity(v) => (0, v.clone()),
        BoundaryCondition::Pressure(p) => (1, p.clone()),
        BoundaryCondition::VelocityWithAdmittance { velocity, .. } => (0, velocity.clone()),
        BoundaryCondition::TransferAdmittance { .. }
        | BoundaryCondition::TransferWithSurfaceAdmittance { .. } => {
            (0, vec![Complex64::new(0.0, 0.0)])
        }
    }
}

/// Get element node coordinates as a DenseMatrix.
pub fn get_element_coords(element: &Element, nodes: &[[f64; 3]]) -> DenseMatrix<f64> {
    let num_nodes = element.connectivity.len();
    let mut coords = DenseMatrix::<f64>::zeros(num_nodes, 3);

    for (i, &node_idx) in element.connectivity.iter().enumerate() {
        for j in 0..3 {
            coords[[i, j]] = nodes[node_idx][j];
        }
    }

    coords
}

/// Compute a near-field block between two sets of constant panels.
///
/// Flat geometric normals are used consistently with the panel geometry,
/// surface measure, and jump terms.
///
/// Used by SLFMM and MLFMM near-field assembly.
pub fn compute_near_block_stabilized(
    elements: &[Element],
    all_coords: &[DenseMatrix<f64>],
    source_indices: &[usize],
    field_indices: &[usize],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    is_self: bool,
    sign: f64,
) -> DenseMatrix<Complex64> {
    let n_source = source_indices.len();
    let n_field = field_indices.len();
    let mut block = DenseMatrix::<Complex64>::zeros(n_source, n_field);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    for (i, &src_idx) in source_indices.iter().enumerate() {
        let source_elem = &elements[src_idx];
        if source_elem.property.is_evaluation() {
            continue;
        }

        let source_normal = source_elem.normal;

        for (j, &fld_idx) in field_indices.iter().enumerate() {
            let field_elem = &elements[fld_idx];
            if field_elem.property.is_evaluation() {
                continue;
            }
            let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);

            let element_coords = &all_coords[fld_idx];

            let result = if is_self && src_idx == fld_idx {
                // Singular integration
                singular_integration(
                    &source_elem.center,
                    &source_normal,
                    element_coords,
                    field_elem.element_type,
                    physics,
                    None,
                    0,
                    false,
                )
            } else {
                // Regular integration
                regular_integration(
                    &source_elem.center,
                    &source_normal,
                    element_coords,
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    None,
                    0,
                    false,
                )
            };

            let mut coeff = match field_bc_type {
                0 => {
                    sign * (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta)
                }
                1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                _ => Complex64::new(0.0, 0.0),
            };
            if field_bc_type == 0
                && let Some(admittance) = field_elem.boundary_condition.admittance()
            {
                coeff -= sign
                    * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                    * admittance
                    * i_omega_rho;
            }

            if is_self && src_idx == fld_idx {
                coeff -= sign * (gamma * 0.5);
                if let Some(admittance) = source_elem.boundary_condition.admittance() {
                    coeff -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                }
            }

            block[[i, j]] = coeff;
        }
    }

    block
}

#[cfg(test)]
mod fmm_accuracy_tests {
    use super::*;

    fn probe() -> Vec<Complex64> {
        vec![Complex64::new(1.0, 0.2), Complex64::new(-0.4, 0.7)]
    }

    #[test]
    fn checked_fmm_accuracy_accepts_all_actions_and_rejects_malformed_data() {
        let probes = vec![probe()];
        let references = probes.clone();
        for action in [
            FmmAction::Forward,
            FmmAction::Transpose,
            FmmAction::Hermitian,
        ] {
            assert_eq!(
                checked_fmm_relative_action_error(action, &probes, &references, 2, |_, input| {
                    input.to_vec()
                },),
                Ok(0.0)
            );
        }

        assert!(matches!(
            checked_fmm_relative_action_error(FmmAction::Forward, &[], &[], 2, |_, input| input
                .to_vec(),),
            Err(FmmAccuracyError::EmptyProbeSet)
        ));
        assert!(matches!(
            checked_fmm_relative_action_error(FmmAction::Forward, &probes, &[], 2, |_, input| {
                input.to_vec()
            },),
            Err(FmmAccuracyError::ProbeCountMismatch { .. })
        ));

        let short_probe = vec![vec![Complex64::new(1.0, 0.0)]];
        assert!(matches!(
            checked_fmm_relative_action_error(
                FmmAction::Forward,
                &short_probe,
                &short_probe,
                2,
                |_, input| input.to_vec(),
            ),
            Err(FmmAccuracyError::InputLength { .. })
        ));

        let non_finite_probe = vec![vec![
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(0.0, 0.0),
        ]];
        assert!(matches!(
            checked_fmm_relative_action_error(
                FmmAction::Forward,
                &non_finite_probe,
                &references,
                2,
                |_, input| input.to_vec(),
            ),
            Err(FmmAccuracyError::NonFiniteInput { .. })
        ));

        let short_reference = vec![vec![Complex64::new(1.0, 0.0)]];
        assert!(matches!(
            checked_fmm_relative_action_error(
                FmmAction::Forward,
                &probes,
                &short_reference,
                2,
                |_, input| input.to_vec(),
            ),
            Err(FmmAccuracyError::ReferenceLength { .. })
        ));

        let non_finite_reference = vec![vec![
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, 0.0),
        ]];
        assert!(matches!(
            checked_fmm_relative_action_error(
                FmmAction::Forward,
                &probes,
                &non_finite_reference,
                2,
                |_, input| input.to_vec(),
            ),
            Err(FmmAccuracyError::NonFiniteReference { .. })
        ));

        assert!(matches!(
            checked_fmm_relative_action_error(
                FmmAction::Forward,
                &probes,
                &references,
                2,
                |_, _| vec![Complex64::new(0.0, 0.0)],
            ),
            Err(FmmAccuracyError::OutputLength { .. })
        ));

        assert!(matches!(
            checked_fmm_relative_action_error(
                FmmAction::Forward,
                &probes,
                &references,
                2,
                |_, _| vec![Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0)],
            ),
            Err(FmmAccuracyError::NonFiniteOutput { .. })
        ));
    }
}

/// Assemble a near-field block for one-proxy-per-basis-DOF layouts.
///
/// Proxy elements retain the original panel geometry and boundary condition,
/// while `group` identifies the panel basis order and local row/column slot.
/// The returned block therefore has one row and one column per proxy, but the
/// field integration is weighted by the selected basis function. This is the
/// dense near-field counterpart of the basis-weighted FMM multipoles.
#[allow(clippy::too_many_arguments)]
pub fn compute_near_block_stabilized_with_basis(
    elements: &[Element],
    all_coords: &[DenseMatrix<f64>],
    source_indices: &[usize],
    field_indices: &[usize],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    is_self: bool,
    sign: f64,
) -> DenseMatrix<Complex64> {
    let mut block = DenseMatrix::<Complex64>::zeros(source_indices.len(), field_indices.len());
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    for (row, &source_index) in source_indices.iter().enumerate() {
        let source_elem = &elements[source_index];
        let source_dof = source_elem.dof_addresses.first().copied();
        let source_point = source_elem.center;
        let source_normal = source_elem.normal;

        for (col, &field_index) in field_indices.iter().enumerate() {
            let field_elem = &elements[field_index];
            let Some(field_coords) = all_coords.get(field_index) else {
                block[[row, col]] = Complex64::new(f64::NAN, f64::NAN);
                continue;
            };
            let Some(_field_basis) = basis_order_for_proxy(field_elem) else {
                block[[row, col]] = Complex64::new(f64::NAN, f64::NAN);
                continue;
            };
            let field_basis_count = decode_dof_proxy_group(field_elem.group)
                .map(|(count, _)| count)
                .unwrap_or_else(|| field_elem.dof_addresses.len().max(1));
            let field_local = proxy_local_index(field_elem);
            if field_local >= field_basis_count {
                block[[row, col]] = Complex64::new(f64::NAN, f64::NAN);
                continue;
            }

            let curved = field_elem
                .node_normals
                .iter()
                .any(|normal| normal.iter().any(|value| value.abs() > 1e-15));
            let same_panel = is_self && source_elem.connectivity == field_elem.connectivity;
            let result = if same_panel {
                if curved {
                    singular_integration_linear_curved(
                        &source_point,
                        &source_normal,
                        field_coords,
                        field_elem.element_type,
                        physics,
                        field_basis_count,
                        Some(field_elem.node_normals.as_slice()),
                    )
                } else {
                    singular_integration_linear(
                        &source_point,
                        &source_normal,
                        field_coords,
                        field_elem.element_type,
                        physics,
                        field_basis_count,
                    )
                }
            } else if curved {
                regular_integration_linear_with_normals(
                    &source_point,
                    &source_normal,
                    field_coords,
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    field_basis_count,
                    Some(field_elem.node_normals.as_slice()),
                )
            } else {
                regular_integration_linear(
                    &source_point,
                    &source_normal,
                    field_coords,
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    field_basis_count,
                )
            };
            let Some(result) = result.basis_results.get(field_local) else {
                block[[row, col]] = Complex64::new(f64::NAN, f64::NAN);
                continue;
            };
            let mut coefficient = match get_bc_type_and_value(&field_elem.boundary_condition).0 {
                0 => {
                    sign * (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta)
                }
                1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                _ => Complex64::new(0.0, 0.0),
            };
            if let Some(admittance) = field_elem.boundary_condition.admittance() {
                coefficient -= sign
                    * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                    * admittance
                    * i_omega_rho;
            }

            // The free Burton--Miller trace is diagonal in the global DOF
            // layout. Repeated continuous proxies intentionally contribute
            // once per panel incidence, matching the H-matrix expansion.
            if same_panel
                && let (Some(source_dof), Some(field_dof)) =
                    (source_dof, field_elem.dof_addresses.first().copied())
                && source_dof == field_dof
            {
                match get_bc_type_and_value(&field_elem.boundary_condition).0 {
                    0 => {
                        coefficient -= sign * gamma * 0.5;
                        if let Some(admittance) = field_elem.boundary_condition.admittance() {
                            coefficient -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                        }
                    }
                    1 => coefficient += sign * beta * tau * 0.5,
                    _ => {}
                }
            }
            block[[row, col]] = coefficient;
        }
    }
    block
}
