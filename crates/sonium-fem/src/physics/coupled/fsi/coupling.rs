//! Energy-conserving FSI interface transfer and partitioned relaxation.

use crate::assembly::TripletMatrix;

/// One scalar fluid/solid velocity constraint with quadrature weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsiInterfaceDofPair {
    pub fluid_dof: usize,
    pub solid_dof: usize,
    pub weight: f64,
}

/// Three-block `(fluid velocity, solid velocity, multiplier)` interface system.
pub struct FsiInterfaceConstraint {
    pub matrix: TripletMatrix<f64>,
    pub fluid_dofs: usize,
    pub solid_dofs: usize,
    pub multiplier_dofs: usize,
}

/// Assemble the symmetric saddle operator
///
/// ```text
/// [ 0  0   C_f^T ]
/// [ 0  0  -C_s^T ]
/// [ C_f -C_s  0   ]
/// ```
///
/// It enforces `u_f = v_s`; multiplier work cancels exactly between domains.
pub fn assemble_fsi_interface_constraint(
    fluid_dofs: usize,
    solid_dofs: usize,
    pairs: &[FsiInterfaceDofPair],
) -> FsiInterfaceConstraint {
    let multiplier_dofs = pairs.len();
    let multiplier_offset = fluid_dofs + solid_dofs;
    let mut matrix = TripletMatrix::with_capacity(
        fluid_dofs + solid_dofs + multiplier_dofs,
        fluid_dofs + solid_dofs + multiplier_dofs,
        4 * pairs.len(),
    );
    for (multiplier, pair) in pairs.iter().enumerate() {
        assert!(pair.fluid_dof < fluid_dofs && pair.solid_dof < solid_dofs);
        assert!(pair.weight.is_finite() && pair.weight > 0.0);
        let lambda = multiplier_offset + multiplier;
        let solid = fluid_dofs + pair.solid_dof;
        matrix.push(pair.fluid_dof, lambda, pair.weight);
        matrix.push(solid, lambda, -pair.weight);
        matrix.push(lambda, pair.fluid_dof, pair.weight);
        matrix.push(lambda, solid, -pair.weight);
    }
    FsiInterfaceConstraint {
        matrix,
        fluid_dofs,
        solid_dofs,
        multiplier_dofs,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceTractionTransfer {
    pub fluid_forces: Vec<f64>,
    pub solid_forces: Vec<f64>,
    pub net_force: f64,
}

/// Transfer multiplier tractions with exact discrete action-reaction balance.
pub fn transfer_interface_tractions(
    fluid_dofs: usize,
    solid_dofs: usize,
    pairs: &[FsiInterfaceDofPair],
    multipliers: &[f64],
) -> InterfaceTractionTransfer {
    assert_eq!(pairs.len(), multipliers.len());
    let mut fluid_forces = vec![0.0; fluid_dofs];
    let mut solid_forces = vec![0.0; solid_dofs];
    for (pair, multiplier) in pairs.iter().zip(multipliers) {
        let force = pair.weight * multiplier;
        fluid_forces[pair.fluid_dof] += force;
        solid_forces[pair.solid_dof] -= force;
    }
    let net_force = fluid_forces.iter().sum::<f64>() + solid_forces.iter().sum::<f64>();
    InterfaceTractionTransfer {
        fluid_forces,
        solid_forces,
        net_force,
    }
}

/// Stateful vector Aitken delta-squared relaxation.
#[derive(Debug, Clone)]
pub struct AitkenRelaxation {
    relaxation: f64,
    minimum: f64,
    maximum: f64,
    previous_residual: Option<Vec<f64>>,
}

impl AitkenRelaxation {
    pub fn new(initial: f64, minimum: f64, maximum: f64) -> Self {
        assert!(0.0 < minimum && minimum <= initial && initial <= maximum && maximum <= 1.0);
        Self {
            relaxation: initial,
            minimum,
            maximum,
            previous_residual: None,
        }
    }

    pub fn relaxation(&self) -> f64 {
        self.relaxation
    }

    pub fn update(&mut self, previous_iterate: &[f64], unrelaxed_iterate: &[f64]) -> Vec<f64> {
        assert_eq!(previous_iterate.len(), unrelaxed_iterate.len());
        let residual: Vec<_> = unrelaxed_iterate
            .iter()
            .zip(previous_iterate)
            .map(|(new, old)| new - old)
            .collect();
        if let Some(previous) = &self.previous_residual {
            let delta: Vec<_> = residual
                .iter()
                .zip(previous)
                .map(|(current, old)| current - old)
                .collect();
            let denominator = delta.iter().map(|value| value * value).sum::<f64>();
            if denominator > f64::MIN_POSITIVE {
                let numerator = previous
                    .iter()
                    .zip(&delta)
                    .map(|(old, change)| old * change)
                    .sum::<f64>();
                self.relaxation =
                    (-self.relaxation * numerator / denominator).clamp(self.minimum, self.maximum);
            }
        }
        self.previous_residual = Some(residual.clone());
        previous_iterate
            .iter()
            .zip(residual)
            .map(|(old, change)| old + self.relaxation * change)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interface_constraint_is_symmetric_and_traction_is_conservative() {
        let pairs = [
            FsiInterfaceDofPair {
                fluid_dof: 0,
                solid_dof: 1,
                weight: 2.0,
            },
            FsiInterfaceDofPair {
                fluid_dof: 1,
                solid_dof: 0,
                weight: 3.0,
            },
        ];
        let constraint = assemble_fsi_interface_constraint(2, 2, &pairs);
        let csr = constraint.matrix.to_csr();
        for row in 0..csr.num_rows {
            for (column, value) in csr.row_entries(row) {
                let transpose = csr
                    .row_entries(column)
                    .find(|(candidate, _)| *candidate == row)
                    .map(|(_, candidate)| candidate)
                    .unwrap_or(0.0);
                assert_eq!(value, transpose);
            }
        }
        let transfer = transfer_interface_tractions(2, 2, &pairs, &[5.0, -7.0]);
        assert_eq!(transfer.net_force, 0.0);
    }

    #[test]
    fn aitken_relaxation_stays_in_configured_bounds() {
        let mut aitken = AitkenRelaxation::new(0.5, 0.1, 0.9);
        let first = aitken.update(&[0.0, 0.0], &[1.0, -1.0]);
        let _second = aitken.update(&first, &[0.25, -0.25]);
        assert!((0.1..=0.9).contains(&aitken.relaxation()));
    }
}
