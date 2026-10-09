//! Small conjugate-style thermal network for speaker operating points.

use serde::{Deserialize, Serialize};

use crate::{ModelingError, require_finite, require_nonnegative, require_positive};

pub const THERMAL_NODE_COUNT: usize = 4;

/// Fixed node ordering used by all serialized thermal network artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalNode {
    VoiceCoil = 0,
    Magnet = 1,
    Gap = 2,
    Cabinet = 3,
}

/// Heat generated in each thermal node (W).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalNetworkPower {
    pub voice_coil_w: f64,
    pub magnet_w: f64,
    pub gap_w: f64,
    pub cabinet_w: f64,
}

impl ThermalNetworkPower {
    pub fn as_array(self) -> [f64; THERMAL_NODE_COUNT] {
        [self.voice_coil_w, self.magnet_w, self.gap_w, self.cabinet_w]
    }

    pub fn validate(self) -> Result<(), ModelingError> {
        for (field, value) in [
            ("thermal_power.voice_coil_w", self.voice_coil_w),
            ("thermal_power.magnet_w", self.magnet_w),
            ("thermal_power.gap_w", self.gap_w),
            ("thermal_power.cabinet_w", self.cabinet_w),
        ] {
            require_nonnegative(field, value)?;
        }
        Ok(())
    }
}

/// Conductance/capacity specification for the coil, magnet, gap and cabinet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalNetworkSpec {
    pub ambient_temperature_k: f64,
    pub heat_capacity_j_per_k: [f64; THERMAL_NODE_COUNT],
    pub ambient_conductance_w_per_k: [f64; THERMAL_NODE_COUNT],
    pub inter_node_conductance_w_per_k: [[f64; THERMAL_NODE_COUNT]; THERMAL_NODE_COUNT],
}

impl Default for ThermalNetworkSpec {
    fn default() -> Self {
        Self {
            ambient_temperature_k: 293.15,
            heat_capacity_j_per_k: [12.0, 120.0, 20.0, 500.0],
            ambient_conductance_w_per_k: [0.33, 0.15, 0.25, 1.0],
            inter_node_conductance_w_per_k: [
                [0.0, 0.05, 0.20, 0.02],
                [0.05, 0.0, 0.10, 0.04],
                [0.20, 0.10, 0.0, 0.08],
                [0.02, 0.04, 0.08, 0.0],
            ],
        }
    }
}

impl ThermalNetworkSpec {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive(
            "thermal_network.ambient_temperature_k",
            self.ambient_temperature_k,
        )?;
        for (index, value) in self.heat_capacity_j_per_k.iter().enumerate() {
            require_positive("thermal_network.heat_capacity_j_per_k", *value).map_err(|_| {
                ModelingError::OutOfRange {
                    field: "thermal_network.heat_capacity_j_per_k",
                    constraint: if index < THERMAL_NODE_COUNT {
                        "positive for every node"
                    } else {
                        "positive"
                    },
                }
            })?;
        }
        for value in self.ambient_conductance_w_per_k {
            require_positive("thermal_network.ambient_conductance_w_per_k", value)?;
        }
        for row in self.inter_node_conductance_w_per_k {
            for value in row {
                require_nonnegative("thermal_network.inter_node_conductance_w_per_k", value)?;
            }
        }
        for i in 0..THERMAL_NODE_COUNT {
            if self.inter_node_conductance_w_per_k[i][i] != 0.0 {
                return Err(ModelingError::OutOfRange {
                    field: "thermal_network.inter_node_conductance_w_per_k",
                    constraint: "zero diagonal",
                });
            }
            for j in (i + 1)..THERMAL_NODE_COUNT {
                if (self.inter_node_conductance_w_per_k[i][j]
                    - self.inter_node_conductance_w_per_k[j][i])
                    .abs()
                    > 1.0e-12
                {
                    return Err(ModelingError::OutOfRange {
                        field: "thermal_network.inter_node_conductance_w_per_k",
                        constraint: "symmetric reciprocal conductances",
                    });
                }
            }
        }
        Ok(())
    }

    /// Solve the steady-state conjugate network with ambient temperature as
    /// the Dirichlet reference.  The returned temperatures and residual are
    /// suitable for a power-compression or demagnetization constraint.
    pub fn solve_steady_state(
        self,
        power: ThermalNetworkPower,
    ) -> Result<ThermalNetworkState, ModelingError> {
        self.validate()?;
        power.validate()?;
        let mut matrix = [[0.0; THERMAL_NODE_COUNT]; THERMAL_NODE_COUNT];
        let mut rhs = power.as_array();
        for i in 0..THERMAL_NODE_COUNT {
            matrix[i][i] = self.ambient_conductance_w_per_k[i];
            rhs[i] += self.ambient_conductance_w_per_k[i] * self.ambient_temperature_k;
            for j in 0..THERMAL_NODE_COUNT {
                let conductance = self.inter_node_conductance_w_per_k[i][j];
                matrix[i][i] += conductance;
                matrix[i][j] -= conductance;
            }
        }
        let temperatures = solve_thermal_system(matrix, rhs)?;
        let mut residual: f64 = 0.0;
        for i in 0..THERMAL_NODE_COUNT {
            let mut balance = self.ambient_conductance_w_per_k[i]
                * (temperatures[i] - self.ambient_temperature_k);
            for j in 0..THERMAL_NODE_COUNT {
                balance +=
                    self.inter_node_conductance_w_per_k[i][j] * (temperatures[i] - temperatures[j]);
            }
            residual = residual.max((balance - power.as_array()[i]).abs());
        }
        Ok(ThermalNetworkState {
            temperatures_k: temperatures,
            power_w: power,
            maximum_balance_residual_w: residual,
        })
    }

    /// Advance the network by one unconditionally stable implicit-Euler step.
    /// This transient counterpart uses node capacities rather than treating
    /// the thermal model as a static copper-only correction.
    pub fn step_implicit(
        self,
        current_temperatures_k: [f64; THERMAL_NODE_COUNT],
        power: ThermalNetworkPower,
        time_step_s: f64,
    ) -> Result<ThermalNetworkState, ModelingError> {
        self.validate()?;
        power.validate()?;
        require_positive("thermal_network.time_step_s", time_step_s)?;
        for temperature in current_temperatures_k {
            require_positive("thermal_network.current_temperature_k", temperature)?;
        }
        let mut matrix = [[0.0; THERMAL_NODE_COUNT]; THERMAL_NODE_COUNT];
        let mut rhs = power.as_array();
        for i in 0..THERMAL_NODE_COUNT {
            let capacity_rate = self.heat_capacity_j_per_k[i] / time_step_s;
            matrix[i][i] = capacity_rate + self.ambient_conductance_w_per_k[i];
            rhs[i] += capacity_rate * current_temperatures_k[i]
                + self.ambient_conductance_w_per_k[i] * self.ambient_temperature_k;
            for j in 0..THERMAL_NODE_COUNT {
                let conductance = self.inter_node_conductance_w_per_k[i][j];
                matrix[i][i] += conductance;
                matrix[i][j] -= conductance;
            }
        }
        let temperatures = solve_thermal_system(matrix, rhs)?;
        let mut residual: f64 = 0.0;
        let temperatures_array = temperatures;
        let powers = power.as_array();
        for i in 0..THERMAL_NODE_COUNT {
            let capacity_rate = self.heat_capacity_j_per_k[i] / time_step_s;
            let mut balance = capacity_rate * (temperatures_array[i] - current_temperatures_k[i]);
            balance += self.ambient_conductance_w_per_k[i]
                * (temperatures_array[i] - self.ambient_temperature_k);
            for j in 0..THERMAL_NODE_COUNT {
                balance += self.inter_node_conductance_w_per_k[i][j]
                    * (temperatures_array[i] - temperatures_array[j]);
            }
            residual = residual.max((balance - powers[i]).abs());
        }
        Ok(ThermalNetworkState {
            temperatures_k: temperatures_array,
            power_w: power,
            maximum_balance_residual_w: residual,
        })
    }
}

/// Temperatures and closure evidence from a thermal-network solve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalNetworkState {
    pub temperatures_k: [f64; THERMAL_NODE_COUNT],
    pub power_w: ThermalNetworkPower,
    pub maximum_balance_residual_w: f64,
}

/// Driver-owned policy for promoting electrical/magnetic losses into the
/// conjugate thermal network used by the product simulator.
///
/// The network itself is deliberately generic; this policy supplies the
/// source split that the driver model can actually identify. Eddy-current
/// loss is divided between magnet and gap nodes, while cabinet heating is an
/// explicit external source. The coil temperature is fed back into copper
/// resistance using the configured temperature coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriverThermalNetworkConfig {
    pub specification: ThermalNetworkSpec,
    /// Fraction of eddy-current loss assigned to the magnet node. The
    /// remainder is assigned to the gap node.
    pub eddy_loss_to_magnet_fraction: f64,
    /// Heat entering the cabinet node from non-electrical sources (W), such
    /// as panel/brace loss or an adjacent driver enclosure model.
    pub cabinet_heat_w: f64,
    /// Fractional copper-resistance change per kelvin.
    pub copper_resistance_per_k: f64,
}

impl Default for DriverThermalNetworkConfig {
    fn default() -> Self {
        Self {
            specification: ThermalNetworkSpec::default(),
            eddy_loss_to_magnet_fraction: 0.5,
            cabinet_heat_w: 0.0,
            copper_resistance_per_k: 0.00393,
        }
    }
}

impl DriverThermalNetworkConfig {
    pub fn validate(self) -> Result<(), ModelingError> {
        self.specification.validate()?;
        require_finite(
            "driver_thermal.eddy_loss_to_magnet_fraction",
            self.eddy_loss_to_magnet_fraction,
        )?;
        if !(0.0..=1.0).contains(&self.eddy_loss_to_magnet_fraction) {
            return Err(ModelingError::OutOfRange {
                field: "driver_thermal.eddy_loss_to_magnet_fraction",
                constraint: "between zero and one",
            });
        }
        require_nonnegative("driver_thermal.cabinet_heat_w", self.cabinet_heat_w)?;
        require_finite(
            "driver_thermal.copper_resistance_per_k",
            self.copper_resistance_per_k,
        )?;
        Ok(())
    }

    /// Convert the loss channels reported by the electromechanical solve into
    /// the fixed thermal-node ordering.
    pub fn power_from_losses(
        self,
        copper_loss_w: f64,
        eddy_loss_w: f64,
    ) -> Result<ThermalNetworkPower, ModelingError> {
        self.validate()?;
        require_nonnegative("driver_thermal.copper_loss_w", copper_loss_w)?;
        require_nonnegative("driver_thermal.eddy_loss_w", eddy_loss_w)?;
        let magnet = eddy_loss_w * self.eddy_loss_to_magnet_fraction;
        let gap = eddy_loss_w - magnet;
        let power = ThermalNetworkPower {
            voice_coil_w: copper_loss_w,
            magnet_w: magnet,
            gap_w: gap,
            cabinet_w: self.cabinet_heat_w,
        };
        power.validate()?;
        Ok(power)
    }

    /// Compute the hot copper resistance from the solved coil temperature.
    pub fn copper_resistance_at(
        self,
        cold_resistance_ohm: f64,
        coil_temperature_k: f64,
    ) -> Result<f64, ModelingError> {
        self.validate()?;
        require_positive("driver_thermal.cold_resistance_ohm", cold_resistance_ohm)?;
        require_positive("driver_thermal.coil_temperature_k", coil_temperature_k)?;
        let factor = 1.0
            + self.copper_resistance_per_k
                * (coil_temperature_k - self.specification.ambient_temperature_k);
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "driver_thermal.copper_resistance_per_k",
                constraint: "must preserve positive copper resistance",
            });
        }
        Ok(cold_resistance_ohm * factor)
    }
}

fn solve_thermal_system(
    mut matrix: [[f64; THERMAL_NODE_COUNT]; THERMAL_NODE_COUNT],
    mut rhs: [f64; THERMAL_NODE_COUNT],
) -> Result<[f64; THERMAL_NODE_COUNT], ModelingError> {
    for pivot in 0..THERMAL_NODE_COUNT {
        let row = (pivot..THERMAL_NODE_COUNT)
            .max_by(|&left, &right| matrix[left][pivot].total_cmp(&matrix[right][pivot]))
            .unwrap();
        if matrix[row][pivot] <= f64::MIN_POSITIVE {
            return Err(ModelingError::OutOfRange {
                field: "thermal_network",
                constraint: "non-singular conductance matrix",
            });
        }
        if row != pivot {
            matrix.swap(row, pivot);
            rhs.swap(row, pivot);
        }
        let pivot_row = matrix[pivot];
        let pivot_rhs = rhs[pivot];
        for (row_values, rhs_value) in matrix.iter_mut().zip(rhs.iter_mut()).skip(pivot + 1) {
            let factor = row_values[pivot] / pivot_row[pivot];
            for (column, value) in row_values.iter_mut().enumerate().skip(pivot) {
                *value -= factor * pivot_row[column];
            }
            *rhs_value -= factor * pivot_rhs;
        }
    }
    let mut solution = [0.0; THERMAL_NODE_COUNT];
    for row in (0..THERMAL_NODE_COUNT).rev() {
        let tail = ((row + 1)..THERMAL_NODE_COUNT)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
        require_finite("thermal_network.temperature_k", solution[row])?;
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_state_heating_reaches_all_coupled_nodes_and_closes() {
        let state = ThermalNetworkSpec::default()
            .solve_steady_state(ThermalNetworkPower {
                voice_coil_w: 24.0,
                magnet_w: 1.0,
                gap_w: 0.5,
                cabinet_w: 0.0,
            })
            .unwrap();
        assert!(
            state.temperatures_k[ThermalNode::VoiceCoil as usize]
                > state.temperatures_k[ThermalNode::Cabinet as usize]
        );
        assert!(state.maximum_balance_residual_w < 1.0e-10);
    }

    #[test]
    fn nonreciprocal_conductance_is_rejected() {
        let mut spec = ThermalNetworkSpec::default();
        spec.inter_node_conductance_w_per_k[0][1] = 0.4;
        assert!(spec.validate().is_err());
    }

    #[test]
    fn implicit_step_uses_node_capacity_and_moves_toward_steady_state() {
        let spec = ThermalNetworkSpec::default();
        let initial = [spec.ambient_temperature_k; THERMAL_NODE_COUNT];
        let power = ThermalNetworkPower {
            voice_coil_w: 24.0,
            magnet_w: 0.0,
            gap_w: 0.0,
            cabinet_w: 0.0,
        };
        let state = spec.step_implicit(initial, power, 0.1).unwrap();
        let steady = spec.solve_steady_state(power).unwrap();
        assert!(state.temperatures_k[ThermalNode::VoiceCoil as usize] > initial[0]);
        assert!(
            state.temperatures_k[ThermalNode::VoiceCoil as usize]
                < steady.temperatures_k[ThermalNode::VoiceCoil as usize]
        );
        assert!(state.maximum_balance_residual_w < 1.0e-10);
    }

    #[test]
    fn driver_policy_splits_losses_and_feeds_coil_temperature_back() {
        let policy = DriverThermalNetworkConfig {
            eddy_loss_to_magnet_fraction: 0.25,
            cabinet_heat_w: 2.0,
            ..Default::default()
        };
        let power = policy.power_from_losses(4.0, 8.0).unwrap();
        assert_eq!(power.voice_coil_w, 4.0);
        assert_eq!(power.magnet_w, 2.0);
        assert_eq!(power.gap_w, 6.0);
        assert_eq!(power.cabinet_w, 2.0);
        let state = policy.specification.solve_steady_state(power).unwrap();
        let hot = policy
            .copper_resistance_at(6.0, state.temperatures_k[ThermalNode::VoiceCoil as usize])
            .unwrap();
        assert!(hot > 6.0);
    }

    #[test]
    fn driver_policy_rejects_invalid_loss_split() {
        let policy = DriverThermalNetworkConfig {
            eddy_loss_to_magnet_fraction: 1.1,
            ..Default::default()
        };
        assert!(policy.validate().is_err());
    }
}
