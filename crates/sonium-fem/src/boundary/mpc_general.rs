//! General multipoint constraints for structural coupling.
//!
//! Real-valued builders over [`MpcSystem`](super::periodic::MpcSystem) cover
//! tied interfaces, beam-shell-solid distribution coupling, and rigid links
//! without touching the periodic-BC machinery. Dense condensation helpers
//! support QA-sized systems directly.

// Rust guideline compliant 2026-02-21

use super::periodic::{MpcError, MpcSystem, MpcTerm, MultiPointConstraint};
use num_complex::Complex64;

/// Build one real constraint `u_slave = sum(coeff * u_master)`.
///
/// # Panics
///
/// Panics for an empty master list, which [`MpcSystem`] rejects downstream.
#[must_use]
pub fn real_mpc(slave: usize, masters: &[(usize, f64)]) -> MultiPointConstraint {
    assert!(!masters.is_empty(), "MPC needs at least one master");
    MultiPointConstraint {
        slave,
        masters: masters
            .iter()
            .map(|&(dof, coefficient)| MpcTerm {
                dof,
                coefficient: Complex64::new(coefficient, 0.0),
            })
            .collect(),
    }
}

/// Tie one slave DOF to one master DOF with a scale factor.
#[must_use]
pub fn tie_dofs(slave: usize, master: usize, coefficient: f64) -> MultiPointConstraint {
    real_mpc(slave, &[(master, coefficient)])
}

/// Tie every component of a slave node to a master node.
#[must_use]
pub fn tie_node_vectors(
    slave_node: usize,
    master_node: usize,
    dofs_per_node: usize,
) -> Vec<MultiPointConstraint> {
    (0..dofs_per_node)
        .map(|component| {
            tie_dofs(
                slave_node * dofs_per_node + component,
                master_node * dofs_per_node + component,
                1.0,
            )
        })
        .collect()
}

/// Tie matching slave and master node lists component by component.
///
/// # Panics
///
/// Panics when the node lists have different lengths.
#[must_use]
pub fn tie_matching_nodes(
    slaves: &[usize],
    masters: &[usize],
    dofs_per_node: usize,
) -> Vec<MultiPointConstraint> {
    assert_eq!(slaves.len(), masters.len(), "tied lists must match");
    slaves
        .iter()
        .zip(masters.iter())
        .flat_map(|(&slave, &master)| tie_node_vectors(slave, master, dofs_per_node))
        .collect()
}

/// Constrain a slave DOF to an interpolation over master DOFs.
///
/// Beam-shell-solid coupling uses this to distribute beam motion onto shell
/// patch nodes (or condense solid faces onto beam nodes) with caller-supplied
/// shape weights.
#[must_use]
pub fn interpolate_slave(slave: usize, masters: &[(usize, f64)]) -> MultiPointConstraint {
    real_mpc(slave, masters)
}

/// Distribute master motion onto patch slaves with shared weights.
///
/// Each slave DOF follows `u_slave = sum(weight * u_master_row)` with one
/// master row per slave; rows share the weight vector (shape values).
///
/// # Panics
///
/// Panics when slave, master-row, or weight counts disagree.
#[must_use]
pub fn distribute_to_patch(
    slaves: &[usize],
    master_rows: &[Vec<usize>],
    weights: &[f64],
) -> Vec<MultiPointConstraint> {
    assert_eq!(
        slaves.len(),
        master_rows.len(),
        "slave and row counts must match"
    );
    for row in master_rows {
        assert_eq!(row.len(), weights.len(), "row and weight counts must match");
    }
    slaves
        .iter()
        .zip(master_rows.iter())
        .map(|(&slave, row)| {
            let masters: Vec<(usize, f64)> = row
                .iter()
                .zip(weights.iter())
                .map(|(&dof, &w)| (dof, w))
                .collect();
            real_mpc(slave, &masters)
        })
        .collect()
}

/// Extract the real prolongation rows of a condensed system.
///
/// # Errors
///
/// Returns [`MpcError::ComplexCoefficientInRealSystem`] when any expansion
/// coefficient carries an imaginary part.
pub fn real_prolongation(mpc: &MpcSystem) -> Result<Vec<Vec<(usize, f64)>>, MpcError> {
    let mut rows = Vec::with_capacity(mpc.num_full_dofs());
    for full_dof in 0..mpc.num_full_dofs() {
        let mut row = Vec::new();
        for &(reduced, coefficient) in mpc.prolongation_row(full_dof) {
            if coefficient.im.abs() > 1.0e-15 * coefficient.norm().max(1.0) {
                return Err(MpcError::ComplexCoefficientInRealSystem { full_dof });
            }
            row.push((reduced, coefficient.re));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Condense a dense symmetric matrix through `K_reduced = P' K P`.
///
/// # Errors
///
/// Returns [`MpcError::MatrixDimension`] for a wrongly sized matrix, or the
/// complex-coefficient error from [`real_prolongation`].
pub fn condense_dense(mpc: &MpcSystem, dense: &[f64], n: usize) -> Result<Vec<f64>, MpcError> {
    if dense.len() != n * n || n != mpc.num_full_dofs() {
        return Err(MpcError::MatrixDimension {
            rows: dense.len(),
            cols: n,
            expected: mpc.num_full_dofs(),
        });
    }
    let prolongation = real_prolongation(mpc)?;
    let reduced = mpc.num_reduced_dofs();
    // temp = K * P stored per full row over reduced columns.
    let mut temp = vec![0.0; n * reduced];
    for row in 0..n {
        for full_col in 0..n {
            let entry = dense[row * n + full_col];
            if entry == 0.0 {
                continue;
            }
            for &(reduced_col, coefficient) in &prolongation[full_col] {
                temp[row * reduced + reduced_col] += entry * coefficient;
            }
        }
    }
    let mut out = vec![0.0; reduced * reduced];
    for full_row in 0..n {
        for &(reduced_row, row_coeff) in &prolongation[full_row] {
            for reduced_col in 0..reduced {
                out[reduced_row * reduced + reduced_col] +=
                    row_coeff * temp[full_row * reduced + reduced_col];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_interface_condenses_to_master() {
        // Two scalar DOFs tied equal; condensing diag(2, 4) gives 6.
        let constraints = vec![tie_dofs(1, 0, 1.0)];
        let mpc = MpcSystem::new(2, &constraints).unwrap();
        assert_eq!(mpc.num_reduced_dofs(), 1);
        let reduced = condense_dense(&mpc, &[2.0, 0.0, 0.0, 4.0], 2).unwrap();
        assert_eq!(reduced.len(), 1);
        assert!((reduced[0] - 6.0).abs() < 1.0e-12);
        let rhs = mpc.condense_real_rhs(&[1.0, 3.0]).unwrap();
        assert!((rhs[0] - 4.0).abs() < 1.0e-12);
        let full = mpc.recover_real(&[2.0]).unwrap();
        assert!((full[0] - 2.0).abs() < 1.0e-12);
        assert!((full[1] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn distribution_weights_reach_prolongation() {
        let constraints = distribute_to_patch(&[2], &[vec![0, 1]], &[0.25, 0.75]);
        let mpc = MpcSystem::new(3, &constraints).unwrap();
        let prolongation = real_prolongation(&mpc).unwrap();
        assert_eq!(prolongation[2].len(), 2);
        assert!((prolongation[2][0].1 - 0.25).abs() < 1.0e-15);
        assert!((prolongation[2][1].1 - 0.75).abs() < 1.0e-15);
    }
}
