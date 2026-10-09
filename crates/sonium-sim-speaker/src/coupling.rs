//! Contracts for coupled multi-driver radiation loads.
//!
//! The product solver currently has a fixed two-driver mesh, but the physical
//! contract is independent of that implementation detail: a radiation load
//! must be finite, reciprocal, and passive before it is coupled to a driver
//! network.  Keeping the checks here makes the same gate available to future
//! multi-driver and reduced-order paths.

use num_complex::Complex64;

const MIN_PIVOT: f64 = 1.0e-14;

/// A square complex radiation-impedance matrix in the modeling convention.
#[derive(Debug, Clone, PartialEq)]
pub struct RadiationMatrix {
    values: Vec<Vec<Complex64>>,
}

/// Diagnostics produced by [`RadiationMatrix::validate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationMatrixDiagnostics {
    /// Relative defect of reciprocity, `max |Zᵢⱼ-Zⱼᵢ| / scale`.
    pub reciprocity_defect: f64,
    /// Smallest pivot of the Hermitian-part Cholesky/LDLᵀ test.
    pub minimum_hermitian_pivot: f64,
    /// Relative active power for the supplied test velocity, when requested.
    pub active_power_w: Option<f64>,
}

/// Failure of a radiation-matrix physical contract.
#[derive(Debug, Clone, PartialEq)]
pub struct RadiationMatrixError {
    message: String,
}

impl RadiationMatrixError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RadiationMatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RadiationMatrixError {}

/// One electrically driven mechanical port in a coupled driver array.
///
/// The coil impedance may already include a complex crossover/network term;
/// keeping it in the port makes the acoustic solve see the same drive model as
/// the electrical response report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledDriverPort {
    pub coil_impedance_ohm: Complex64,
    pub force_factor_tm: f64,
    pub intrinsic_mechanical_impedance_n_s_per_m: Complex64,
    pub voltage_rms_v: Complex64,
}

/// Result of a multi-driver RMS network solve.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledDriverSolution {
    pub velocities_rms_m_per_s: Vec<Complex64>,
    pub currents_rms_a: Vec<Complex64>,
    pub electrical_impedances_ohm: Vec<Complex64>,
    pub radiation_matrix_diagnostics: RadiationMatrixDiagnostics,
    pub radiation_power_w: f64,
    pub power_balance_residual_w: f64,
}

/// Acceptance tolerances for production multi-driver coupling.
///
/// These are explicit solver policy rather than call-site magic numbers, so
/// reduced-order or deliberately low-order studies can opt into a documented
/// looser configuration without weakening the golden production default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationValidationConfig {
    pub reciprocity_tolerance: f64,
    pub passivity_tolerance: f64,
    pub power_balance_relative_tolerance: f64,
}

impl Default for RadiationValidationConfig {
    fn default() -> Self {
        Self {
            reciprocity_tolerance: 1.0e-6,
            passivity_tolerance: 1.0e-6,
            power_balance_relative_tolerance: 1.0e-8,
        }
    }
}

impl RadiationValidationConfig {
    fn validate(self) -> Result<(), RadiationMatrixError> {
        if !self.reciprocity_tolerance.is_finite()
            || self.reciprocity_tolerance < 0.0
            || !self.passivity_tolerance.is_finite()
            || self.passivity_tolerance < 0.0
            || !self.power_balance_relative_tolerance.is_finite()
            || self.power_balance_relative_tolerance < 0.0
        {
            return Err(RadiationMatrixError::new(
                "radiation validation tolerances must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Solve all driver ports against a reciprocal radiation matrix.
pub fn solve_coupled_driver_network(
    ports: &[CoupledDriverPort],
    radiation: &RadiationMatrix,
) -> Result<CoupledDriverSolution, RadiationMatrixError> {
    solve_coupled_driver_network_with_config(ports, radiation, RadiationValidationConfig::default())
}

/// Solve all driver ports with an explicit reciprocity/passivity/power policy.
pub fn solve_coupled_driver_network_with_config(
    ports: &[CoupledDriverPort],
    radiation: &RadiationMatrix,
    validation: RadiationValidationConfig,
) -> Result<CoupledDriverSolution, RadiationMatrixError> {
    validation.validate()?;
    if ports.is_empty() || ports.len() != radiation.dimension() {
        return Err(RadiationMatrixError::new(
            "driver-port count must match the radiation matrix",
        ));
    }
    let radiation_matrix_diagnostics = radiation.validate(
        validation.reciprocity_tolerance,
        validation.passivity_tolerance,
    )?;
    for port in ports {
        if !port.coil_impedance_ohm.re.is_finite()
            || !port.coil_impedance_ohm.im.is_finite()
            || port.coil_impedance_ohm.norm() <= f64::MIN_POSITIVE
            || port.coil_impedance_ohm.re < 0.0
            || !port.force_factor_tm.is_finite()
            || !port.intrinsic_mechanical_impedance_n_s_per_m.re.is_finite()
            || !port.intrinsic_mechanical_impedance_n_s_per_m.im.is_finite()
            || port.intrinsic_mechanical_impedance_n_s_per_m.re < 0.0
            || !port.voltage_rms_v.re.is_finite()
            || !port.voltage_rms_v.im.is_finite()
        {
            return Err(RadiationMatrixError::new(
                "driver network contains a non-finite, singular, or active port",
            ));
        }
    }

    let n = ports.len();
    let mut matrix = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut rhs = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        matrix[i].copy_from_slice(&radiation.values()[i]);
        matrix[i][i] += ports[i].intrinsic_mechanical_impedance_n_s_per_m
            + Complex64::new(ports[i].force_factor_tm.powi(2), 0.0) / ports[i].coil_impedance_ohm;
        rhs[i] = Complex64::new(ports[i].force_factor_tm, 0.0) * ports[i].voltage_rms_v
            / ports[i].coil_impedance_ohm;
    }
    let velocities = solve_dense_complex_system(&mut matrix, &mut rhs)?;
    let currents: Vec<_> = ports
        .iter()
        .zip(&velocities)
        .map(|(port, velocity)| {
            (port.voltage_rms_v - Complex64::new(port.force_factor_tm, 0.0) * velocity)
                / port.coil_impedance_ohm
        })
        .collect();
    let electrical_impedances: Vec<_> = ports
        .iter()
        .zip(&currents)
        .map(|(port, current)| {
            if current.norm() <= f64::MIN_POSITIVE {
                Err(RadiationMatrixError::new(
                    "driver network current vanished while forming input impedance",
                ))
            } else {
                Ok(port.voltage_rms_v / current)
            }
        })
        .collect::<Result<_, _>>()?;
    let radiation_power = radiation.active_power_w(&velocities)?;
    let input_power: f64 = ports
        .iter()
        .zip(&currents)
        .map(|(port, current)| (port.voltage_rms_v * current.conj()).re)
        .sum();
    let copper_power: f64 = ports
        .iter()
        .zip(&currents)
        .map(|(port, current)| port.coil_impedance_ohm.re * current.norm_sqr())
        .sum();
    let mechanical_power: f64 = ports
        .iter()
        .zip(&velocities)
        .map(|(port, velocity)| {
            port.intrinsic_mechanical_impedance_n_s_per_m.re * velocity.norm_sqr()
        })
        .sum();
    let power_balance_residual_w = input_power - copper_power - mechanical_power - radiation_power;
    let scale = input_power
        .abs()
        .max(copper_power.abs())
        .max(mechanical_power.abs())
        .max(radiation_power.abs())
        .max(1.0e-12);
    if !power_balance_residual_w.is_finite()
        || power_balance_residual_w.abs() > validation.power_balance_relative_tolerance * scale
    {
        return Err(RadiationMatrixError::new(format!(
            "coupled driver power balance failed: residual={power_balance_residual_w:.3e} W, scale={scale:.3e} W"
        )));
    }
    Ok(CoupledDriverSolution {
        velocities_rms_m_per_s: velocities,
        currents_rms_a: currents,
        electrical_impedances_ohm: electrical_impedances,
        radiation_matrix_diagnostics,
        radiation_power_w: radiation_power,
        power_balance_residual_w,
    })
}

fn solve_dense_complex_system(
    matrix: &mut [Vec<Complex64>],
    rhs: &mut [Complex64],
) -> Result<Vec<Complex64>, RadiationMatrixError> {
    let n = rhs.len();
    for pivot in 0..n {
        let row = (pivot..n)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .norm()
                    .total_cmp(&matrix[right][pivot].norm())
            })
            .ok_or_else(|| RadiationMatrixError::new("empty driver network"))?;
        if matrix[row][pivot].norm() <= f64::MIN_POSITIVE {
            return Err(RadiationMatrixError::new(
                "coupled driver network is singular",
            ));
        }
        if row != pivot {
            matrix.swap(row, pivot);
            rhs.swap(row, pivot);
        }
        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            matrix[row][pivot] = Complex64::new(0.0, 0.0);
            let pivot_tail = matrix[pivot][(pivot + 1)..n].to_vec();
            for (value, pivot_value) in matrix[row][(pivot + 1)..n].iter_mut().zip(pivot_tail) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![Complex64::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let tail = ((row + 1)..n)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<Complex64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    if solution
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(RadiationMatrixError::new(
            "coupled driver network returned a non-finite velocity",
        ));
    }
    Ok(solution)
}

impl RadiationMatrix {
    /// Construct a matrix after checking that it is square and finite.
    pub fn new(values: Vec<Vec<Complex64>>) -> Result<Self, RadiationMatrixError> {
        let dimension = values.len();
        if dimension == 0 || values.iter().any(|row| row.len() != dimension) {
            return Err(RadiationMatrixError::new(
                "radiation matrix must be a non-empty square matrix",
            ));
        }
        if values
            .iter()
            .flatten()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(RadiationMatrixError::new(
                "radiation matrix contains a non-finite value",
            ));
        }
        Ok(Self { values })
    }

    /// Borrow the matrix entries in row-major form.
    pub fn values(&self) -> &[Vec<Complex64>] {
        &self.values
    }

    /// Number of coupled driver ports.
    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    /// Relative reciprocity defect using the mechanical-port convention
    /// `Zᵢⱼ = Zⱼᵢ`.
    pub fn reciprocity_defect(&self) -> f64 {
        let scale = self
            .values
            .iter()
            .flatten()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
        let mut defect: f64 = 0.0;
        for i in 0..self.dimension() {
            for j in (i + 1)..self.dimension() {
                defect = defect.max((self.values[i][j] - self.values[j][i]).norm() / scale);
            }
        }
        defect
    }

    /// Smallest diagonal pivot of an LDLᵀ factorization of the Hermitian part
    /// `(Z + Zᴴ)/2`. A negative pivot proves non-passivity. Small negative
    /// values can be admitted by the caller's discretization tolerance.
    pub fn minimum_hermitian_pivot(&self) -> f64 {
        let n = self.dimension();
        let mut lower = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        let mut minimum = f64::INFINITY;
        for i in 0..n {
            let mut diagonal = self.values[i][i].re;
            for value in &lower[i][..i] {
                diagonal -= value.norm_sqr();
            }
            minimum = minimum.min(diagonal);
            if diagonal <= MIN_PIVOT {
                continue;
            }
            let root = diagonal.sqrt();
            let prior_row = lower[i].clone();
            for (j, lower_row) in lower.iter_mut().enumerate().skip(i + 1) {
                let mut value = (self.values[j][i] + self.values[i][j].conj()) * 0.5;
                for (lower_jk, lower_ik) in lower_row[..i].iter().zip(&prior_row[..i]) {
                    value -= *lower_jk * lower_ik.conj();
                }
                lower_row[i] = value / root;
            }
        }
        minimum
    }

    /// Active acoustic power for RMS port velocities, `Re(vᴴ Z v)`.
    pub fn active_power_w(&self, velocities: &[Complex64]) -> Result<f64, RadiationMatrixError> {
        if velocities.len() != self.dimension()
            || velocities
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(RadiationMatrixError::new(
                "velocity vector does not match the radiation matrix",
            ));
        }
        let power = (0..self.dimension())
            .flat_map(|i| (0..self.dimension()).map(move |j| (i, j)))
            .map(|(i, j)| velocities[i].conj() * self.values[i][j] * velocities[j])
            .sum::<Complex64>()
            .re;
        Ok(power)
    }

    /// Enforce reciprocity and positive-semidefinite Hermitian-part gates.
    pub fn validate(
        &self,
        reciprocity_tolerance: f64,
        passivity_tolerance: f64,
    ) -> Result<RadiationMatrixDiagnostics, RadiationMatrixError> {
        if !reciprocity_tolerance.is_finite()
            || reciprocity_tolerance < 0.0
            || !passivity_tolerance.is_finite()
            || passivity_tolerance < 0.0
        {
            return Err(RadiationMatrixError::new(
                "radiation-matrix tolerances must be finite and non-negative",
            ));
        }
        let reciprocity_defect = self.reciprocity_defect();
        if reciprocity_defect > reciprocity_tolerance {
            return Err(RadiationMatrixError::new(format!(
                "radiation matrix reciprocity defect {reciprocity_defect:.3e} exceeds {reciprocity_tolerance:.3e}"
            )));
        }
        let minimum_hermitian_pivot = self.minimum_hermitian_pivot();
        let scale = self
            .values
            .iter()
            .flatten()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(1.0);
        if minimum_hermitian_pivot < -passivity_tolerance * scale {
            return Err(RadiationMatrixError::new(format!(
                "radiation matrix is active: minimum Hermitian pivot {minimum_hermitian_pivot:.3e}"
            )));
        }
        Ok(RadiationMatrixDiagnostics {
            reciprocity_defect,
            minimum_hermitian_pivot,
            active_power_w: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_passive_matrix_passes_and_reports_power() {
        let matrix = RadiationMatrix::new(vec![
            vec![Complex64::new(2.0, 1.0), Complex64::new(0.5, -0.2)],
            vec![Complex64::new(0.5, -0.2), Complex64::new(1.0, 0.3)],
        ])
        .unwrap();
        let diagnostics = matrix.validate(1.0e-12, 1.0e-12).unwrap();
        assert_eq!(diagnostics.reciprocity_defect, 0.0);
        let power = matrix
            .active_power_w(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)])
            .unwrap();
        assert!(power > 0.0);
    }

    #[test]
    fn nonreciprocal_and_active_matrices_are_rejected() {
        let nonreciprocal = RadiationMatrix::new(vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .unwrap();
        assert!(nonreciprocal.validate(1.0e-6, 1.0e-6).is_err());

        let active = RadiationMatrix::new(vec![
            vec![Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .unwrap();
        assert!(active.validate(1.0e-6, 1.0e-6).is_err());
    }

    #[test]
    fn coupled_network_supports_more_than_two_drivers_and_closes_power() {
        let radiation = RadiationMatrix::new(vec![
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.1, 0.0),
                Complex64::new(0.05, -0.01),
            ],
            vec![
                Complex64::new(0.1, 0.0),
                Complex64::new(0.8, 0.1),
                Complex64::new(0.03, 0.0),
            ],
            vec![
                Complex64::new(0.05, -0.01),
                Complex64::new(0.03, 0.0),
                Complex64::new(0.6, 0.05),
            ],
        ])
        .unwrap();
        let ports = vec![
            CoupledDriverPort {
                coil_impedance_ohm: Complex64::new(6.0, 2.0),
                force_factor_tm: 5.0,
                intrinsic_mechanical_impedance_n_s_per_m: Complex64::new(0.4, 1.0),
                voltage_rms_v: Complex64::new(2.0, 0.0),
            },
            CoupledDriverPort {
                coil_impedance_ohm: Complex64::new(4.0, 1.0),
                force_factor_tm: 3.0,
                intrinsic_mechanical_impedance_n_s_per_m: Complex64::new(0.3, 0.7),
                voltage_rms_v: Complex64::new(1.0, 0.2),
            },
            CoupledDriverPort {
                coil_impedance_ohm: Complex64::new(8.0, 3.0),
                force_factor_tm: 2.0,
                intrinsic_mechanical_impedance_n_s_per_m: Complex64::new(0.2, 0.4),
                voltage_rms_v: Complex64::new(0.5, -0.1),
            },
        ];
        let solution = solve_coupled_driver_network(&ports, &radiation).unwrap();
        assert_eq!(solution.velocities_rms_m_per_s.len(), 3);
        assert!(solution.radiation_matrix_diagnostics.reciprocity_defect < 1.0e-12);
        assert!(
            solution
                .radiation_matrix_diagnostics
                .minimum_hermitian_pivot
                > 0.0
        );
        assert!(solution.radiation_power_w > 0.0);
        assert!(solution.power_balance_residual_w.abs() < 1.0e-9);
    }
}
