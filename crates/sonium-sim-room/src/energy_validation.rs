//! Checked active-power balance evidence for RoomSim frequency samples.
//!
//! The ledger accepts powers already integrated from pressure/velocity
//! phasors.  Callers must provide cycle-averaged active power with a consistent
//! outward-normal convention; this API intentionally does not substitute
//! `|p|²/(rho*c)` in a reactive or standing-wave field.  FEM boundary-flux
//! extraction can feed this contract without coupling the contract to one
//! mesh or solver implementation.

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzAssembler;
use std::collections::{BTreeMap, HashMap};

/// One frequency's active acoustic-power ledger, in watts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticPowerBalanceSample {
    /// Strictly increasing frequency associated with this ledger row, in Hz.
    pub frequency_hz: f64,
    /// Time-averaged active power delivered by the source into the domain.
    pub source_power_w: f64,
    /// Non-negative time-averaged power absorbed by passive materials.
    pub absorbed_power_w: f64,
    /// Non-negative time-averaged power leaving through open/radiating bounds.
    pub radiated_power_w: f64,
}

/// Aggregate active-power balance errors over a frequency series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyBalanceReport {
    /// Number of frequency rows checked.
    pub sample_count: usize,
    /// RMS balance residual normalized by source power or the configured floor.
    pub rms_relative_error: f64,
    /// Maximum balance residual normalized by source power or the configured floor.
    pub max_relative_error: f64,
    /// Maximum absolute balance residual, in watts.
    pub max_absolute_error_w: f64,
}

/// Algebraic dissipative-work evidence from one accepted FEM solution.
///
/// These values are intentionally not labelled as watts.  With the room
/// runner's `exp(-iωt)` convention they are the imaginary-part terms in the
/// tested operator identity
///
/// `-Im(pᴴ rhs) = -Im((-k²) pᴴ M p) + Σ -Im(c_b pᴴ M_b p)`.
///
/// A physical power conversion still requires a separately documented source
/// normalization and phasor convention.  Keeping that conversion out of this
/// helper prevents a dimensionally plausible but unverified `W` claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FemOperatorWorkEvidence {
    /// `-Im(pᴴ rhs)`, the work supplied by the assembled source RHS.
    pub source_work: f64,
    /// `-Im((-k²) pᴴ M p)`, the volumetric damping contribution.
    pub volume_damping_work: f64,
    /// Sum of `-Im(c_b pᴴ M_b p)` over active physical boundaries.
    pub boundary_work: f64,
    /// `source_work - volume_damping_work - boundary_work`.
    pub balance_residual: f64,
}

/// Physically normalized active-power evidence from one FEM solution.
///
/// RoomSim's pressure phasors are peak amplitudes and its source RHS is
/// `-iωρQ`.  Consequently, the cycle-average power associated with an
/// operator-work term is
///
/// `P = -Im(pᴴ term) / (2ωρ)`.
///
/// `boundary_power_w` is the signed active power represented by the boundary
/// terms supplied to the assembler.  It may include absorptive walls and
/// open/radiating boundaries; it is deliberately not labelled as radiated
/// power.  A boundary-topology/material classification is required before
/// making that narrower claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FemPhysicalPowerEvidence {
    /// Cycle-average power delivered by the source into the FEM domain (W).
    pub source_power_w: f64,
    /// Cycle-average power represented by complex volumetric damping (W).
    pub volume_damping_power_w: f64,
    /// Signed cycle-average power represented by all active FEM boundary
    /// terms (W); this is not inherently radiated power.
    pub boundary_power_w: f64,
    /// Source power minus volume-damping and boundary power (W).
    pub balance_residual_w: f64,
}

/// Cycle-average active power crossing one FEM impedance boundary (W).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FemBoundaryFluxSample {
    /// Mesh boundary marker used by the FEM assembler.
    pub boundary_tag: usize,
    /// Signed outward active power represented by this boundary term (W).
    pub active_power_w: f64,
}

/// Per-boundary pressure/velocity-flux evidence for one FEM frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct FemBoundaryFluxEvidence {
    /// Tagged boundary contributions in strictly increasing marker order.
    pub samples: Vec<FemBoundaryFluxSample>,
    /// Sum of [`Self::samples`], in watts.
    pub total_power_w: f64,
}

impl FemBoundaryFluxEvidence {
    pub(crate) fn from_tagged_powers(tagged_powers: BTreeMap<usize, f64>) -> Result<Self, String> {
        let samples = tagged_powers
            .into_iter()
            .map(|(boundary_tag, active_power_w)| FemBoundaryFluxSample {
                boundary_tag,
                active_power_w,
            })
            .collect::<Vec<_>>();
        let total_power_w = samples
            .iter()
            .map(|sample| sample.active_power_w)
            .sum::<f64>();
        let evidence = Self {
            samples,
            total_power_w,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.total_power_w.is_finite()
            || self
                .samples
                .iter()
                .any(|sample| !sample.active_power_w.is_finite())
            || self
                .samples
                .windows(2)
                .any(|pair| pair[0].boundary_tag >= pair[1].boundary_tag)
        {
            return Err("FEM boundary-flux evidence contains invalid samples".into());
        }
        let summed_power_w = self
            .samples
            .iter()
            .map(|sample| sample.active_power_w)
            .sum::<f64>();
        let tolerance = 1.0e-12 * self.total_power_w.abs().max(1.0);
        if (summed_power_w - self.total_power_w).abs() > tolerance {
            return Err("FEM boundary-flux total does not match tagged samples".into());
        }
        Ok(())
    }
}

impl FemPhysicalPowerEvidence {
    /// Convert algebraic operator work to cycle-average acoustic power.
    pub fn from_operator_work(
        operator_work: &FemOperatorWorkEvidence,
        frequency_hz: f64,
        density_kg_m3: f64,
    ) -> Result<Self, String> {
        operator_work.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err("FEM physical power frequency must be finite positive".into());
        }
        if !density_kg_m3.is_finite() || density_kg_m3 <= 0.0 {
            return Err("FEM physical power density must be finite positive".into());
        }

        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let normalization = 1.0 / (2.0 * omega * density_kg_m3);
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err("FEM physical power normalization is non-finite".into());
        }

        let evidence = Self {
            source_power_w: operator_work.source_work * normalization,
            volume_damping_power_w: operator_work.volume_damping_work * normalization,
            boundary_power_w: operator_work.boundary_work * normalization,
            balance_residual_w: operator_work.balance_residual * normalization,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if [
            self.source_power_w,
            self.volume_damping_power_w,
            self.boundary_power_w,
            self.balance_residual_w,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err("FEM physical power evidence contains a non-finite value".into());
        }
        Ok(())
    }
}

impl FemOperatorWorkEvidence {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if [
            self.source_work,
            self.volume_damping_work,
            self.boundary_work,
            self.balance_residual,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err("FEM operator-work evidence contains a non-finite value".into());
        }
        Ok(())
    }
}

/// Compute the FEM operator-work ledger for one accepted solution.
///
/// The assembler stores the stiffness, volume-mass, and boundary-mass terms
/// on one aligned CSR pattern.  The stiffness contribution is real-valued and
/// therefore absent from this imaginary-part ledger; all other terms are
/// evaluated directly from the same coefficients used to assemble `A`.
pub(crate) fn fem_operator_work_evidence(
    assembler: &HelmholtzAssembler,
    wavenumber: Complex64,
    boundary_coeffs: &HashMap<usize, Complex64>,
    rhs: &[Complex64],
    solution: &[Complex64],
) -> Result<FemOperatorWorkEvidence, String> {
    if assembler.num_rows == 0
        || rhs.len() != assembler.num_rows
        || solution.len() != assembler.num_rows
        || assembler.row_ptrs.len() != assembler.num_rows + 1
        || assembler.row_ptrs.first() != Some(&0)
        || assembler.row_ptrs.last() != Some(&assembler.col_indices.len())
        || assembler.col_indices.len() != assembler.k_values.len()
        || assembler.col_indices.len() != assembler.m_values.len()
    {
        return Err("FEM operator-work dimensions do not match assembler".into());
    }
    if !wavenumber.re.is_finite() || !wavenumber.im.is_finite() {
        return Err("FEM operator-work wavenumber must be finite".into());
    }
    if assembler
        .row_ptrs
        .windows(2)
        .any(|window| window[0] > window[1] || window[1] > assembler.col_indices.len())
    {
        return Err("FEM operator-work CSR row pointers are invalid".into());
    }
    if assembler
        .col_indices
        .iter()
        .any(|&column| column >= assembler.num_rows)
        || assembler.k_values.iter().any(|value| !value.is_finite())
        || assembler.m_values.iter().any(|value| !value.is_finite())
        || rhs
            .iter()
            .chain(solution)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("FEM operator-work contains invalid vector or column data".into());
    }
    if boundary_coeffs
        .values()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || boundary_coeffs
            .keys()
            .any(|tag| !assembler.boundary_values.contains_key(tag))
        || boundary_coeffs.iter().any(|(tag, _)| {
            assembler.boundary_values.get(tag).is_some_and(|values| {
                values.len() != assembler.col_indices.len()
                    || values.iter().any(|value| !value.is_finite())
            })
        })
    {
        return Err("FEM operator-work contains invalid boundary data".into());
    }

    let mut mass_quadratic = Complex64::new(0.0, 0.0);
    let mut boundary_work = 0.0;
    let mut source_inner = Complex64::new(0.0, 0.0);

    for row in 0..assembler.num_rows {
        source_inner += solution[row].conj() * rhs[row];
        for index in assembler.row_ptrs[row]..assembler.row_ptrs[row + 1] {
            let column = assembler.col_indices[index];
            let basis_product = solution[row].conj() * solution[column];
            mass_quadratic += assembler.m_values[index] * basis_product;

            for (tag, coefficient) in boundary_coeffs {
                if let Some(values) = assembler.boundary_values.get(tag) {
                    boundary_work -= (coefficient * (values[index] * basis_product)).im;
                }
            }
        }
    }

    let volume_operator = -(wavenumber * wavenumber) * mass_quadratic;
    let evidence = FemOperatorWorkEvidence {
        source_work: -source_inner.im,
        volume_damping_work: -volume_operator.im,
        boundary_work,
        balance_residual: -source_inner.im - (-volume_operator.im + boundary_work),
    };
    evidence.validate()?;
    Ok(evidence)
}

/// Extract cycle-average active pressure/velocity flux for every active FEM
/// impedance boundary.
///
/// The runner supplies coefficients of the form `c_Γ = -ikβ`.  Under the
/// `exp(-iωt)` convention, the boundary condition gives the normal velocity
/// phasor and the integrated flux `1/2 Re(p u_n*)` is exactly
/// `-Im(c_Γ pᴴ B_Γ p)/(2ωρ)`.  This is a physical boundary-flux ledger, but it
/// intentionally does not classify a tag as absorption or radiation.
pub(crate) fn fem_boundary_flux_evidence(
    assembler: &HelmholtzAssembler,
    boundary_coeffs: &HashMap<usize, Complex64>,
    frequency_hz: f64,
    density_kg_m3: f64,
    solution: &[Complex64],
) -> Result<FemBoundaryFluxEvidence, String> {
    if assembler.num_rows == 0
        || solution.len() != assembler.num_rows
        || assembler.row_ptrs.len() != assembler.num_rows + 1
        || assembler.row_ptrs.first() != Some(&0)
        || assembler.row_ptrs.last() != Some(&assembler.col_indices.len())
        || assembler.col_indices.len() != assembler.m_values.len()
    {
        return Err("FEM boundary-flux dimensions do not match assembler".into());
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err("FEM boundary-flux frequency must be finite positive".into());
    }
    if !density_kg_m3.is_finite() || density_kg_m3 <= 0.0 {
        return Err("FEM boundary-flux density must be finite positive".into());
    }
    if assembler
        .row_ptrs
        .windows(2)
        .any(|window| window[0] > window[1] || window[1] > assembler.col_indices.len())
        || assembler
            .col_indices
            .iter()
            .any(|&column| column >= assembler.num_rows)
        || solution
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || boundary_coeffs
            .values()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || boundary_coeffs
            .keys()
            .any(|tag| !assembler.boundary_values.contains_key(tag))
        || assembler.boundary_values.values().any(|values| {
            values.len() != assembler.col_indices.len()
                || values.iter().any(|value| !value.is_finite())
        })
    {
        return Err("FEM boundary-flux contains invalid matrix or vector data".into());
    }

    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let normalization = 1.0 / (2.0 * omega * density_kg_m3);
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err("FEM boundary-flux normalization is non-finite".into());
    }

    let mut boundary_work = BTreeMap::<usize, f64>::new();
    for row in 0..assembler.num_rows {
        for index in assembler.row_ptrs[row]..assembler.row_ptrs[row + 1] {
            let column = assembler.col_indices[index];
            let basis_product = solution[row].conj() * solution[column];
            for (&tag, &coefficient) in boundary_coeffs {
                if let Some(values) = assembler.boundary_values.get(&tag) {
                    *boundary_work.entry(tag).or_default() -=
                        (coefficient * (values[index] * basis_product)).im * normalization;
                }
            }
        }
    }

    FemBoundaryFluxEvidence::from_tagged_powers(boundary_work)
}

#[cfg(test)]
mod fem_operator_work_tests {
    use super::*;
    use sonium_fem::assembly::{MassMatrix, StiffnessMatrix};

    fn scalar_assembler() -> HelmholtzAssembler {
        let stiffness = StiffnessMatrix {
            rows: vec![0],
            cols: vec![0],
            values: vec![0.0],
            dim: 1,
        };
        let mass = MassMatrix {
            rows: vec![0],
            cols: vec![0],
            values: vec![1.0],
            dim: 1,
        };
        let boundary = MassMatrix {
            rows: vec![0],
            cols: vec![0],
            values: vec![1.0],
            dim: 1,
        };
        HelmholtzAssembler::from_matrices(&stiffness, &mass, &[(7, boundary)])
    }

    #[test]
    fn operator_work_terms_close_with_explicit_complex_operator() {
        let assembler = scalar_assembler();
        let wavenumber = Complex64::new(1.0, 0.1);
        let boundary_coefficient = Complex64::new(0.0, -0.3);
        let mut boundary_coeffs = HashMap::new();
        boundary_coeffs.insert(7, boundary_coefficient);
        let pressure = Complex64::new(2.0, 0.0);
        let operator = -(wavenumber * wavenumber) + boundary_coefficient;
        let rhs = vec![operator * pressure];

        let evidence =
            fem_operator_work_evidence(&assembler, wavenumber, &boundary_coeffs, &rhs, &[pressure])
                .unwrap();

        assert!((evidence.source_work - 2.0).abs() < 1.0e-12);
        assert!((evidence.volume_damping_work - 0.8).abs() < 1.0e-12);
        assert!((evidence.boundary_work - 1.2).abs() < 1.0e-12);
        assert!(evidence.balance_residual.abs() < 1.0e-12);

        let flux = fem_boundary_flux_evidence(&assembler, &boundary_coeffs, 10.0, 2.0, &[pressure])
            .unwrap();
        assert_eq!(flux.samples.len(), 1);
        assert_eq!(flux.samples[0].boundary_tag, 7);
        let expected_flux = 1.2 / (2.0 * (2.0 * std::f64::consts::PI * 10.0) * 2.0);
        assert!((flux.samples[0].active_power_w - expected_flux).abs() < 1.0e-15);
        assert!((flux.total_power_w - expected_flux).abs() < 1.0e-15);
        let physical_power =
            FemPhysicalPowerEvidence::from_operator_work(&evidence, 10.0, 2.0).unwrap();
        assert!((physical_power.boundary_power_w - flux.total_power_w).abs() < 1.0e-15);
    }

    #[test]
    fn operator_work_rejects_mismatched_or_non_finite_inputs() {
        let assembler = scalar_assembler();
        let boundary_coeffs = HashMap::new();
        let error = fem_operator_work_evidence(
            &assembler,
            Complex64::new(1.0, 0.0),
            &boundary_coeffs,
            &[],
            &[Complex64::new(1.0, 0.0)],
        )
        .unwrap_err();
        assert!(error.contains("dimensions"));

        let error = fem_operator_work_evidence(
            &assembler,
            Complex64::new(f64::NAN, 0.0),
            &boundary_coeffs,
            &[Complex64::new(1.0, 0.0)],
            &[Complex64::new(1.0, 0.0)],
        )
        .unwrap_err();
        assert!(error.contains("wavenumber"));

        let mut unknown_boundary = HashMap::new();
        unknown_boundary.insert(99, Complex64::new(0.0, -0.3));
        let error = fem_operator_work_evidence(
            &assembler,
            Complex64::new(1.0, 0.0),
            &unknown_boundary,
            &[Complex64::new(1.0, 0.0)],
            &[Complex64::new(1.0, 0.0)],
        )
        .unwrap_err();
        assert!(error.contains("boundary"));
    }

    #[test]
    fn physical_power_uses_peak_phasor_cycle_average_normalization() {
        let operator_work = FemOperatorWorkEvidence {
            source_work: 8.0,
            volume_damping_work: 2.0,
            boundary_work: 4.0,
            balance_residual: 2.0,
        };
        let evidence =
            FemPhysicalPowerEvidence::from_operator_work(&operator_work, 10.0, 2.0).unwrap();
        let normalization = 1.0 / (2.0 * (2.0 * std::f64::consts::PI * 10.0) * 2.0);
        assert!((evidence.source_power_w - 8.0 * normalization).abs() < 1.0e-15);
        assert!((evidence.volume_damping_power_w - 2.0 * normalization).abs() < 1.0e-15);
        assert!((evidence.boundary_power_w - 4.0 * normalization).abs() < 1.0e-15);
        assert!((evidence.balance_residual_w - 2.0 * normalization).abs() < 1.0e-15);
    }

    #[test]
    fn physical_power_rejects_invalid_normalization_inputs_and_preserves_sign() {
        let operator_work = FemOperatorWorkEvidence {
            source_work: 1.0,
            volume_damping_work: 0.0,
            boundary_work: -0.5,
            balance_residual: 1.5,
        };
        let evidence =
            FemPhysicalPowerEvidence::from_operator_work(&operator_work, 100.0, 1.2).unwrap();
        assert!(evidence.boundary_power_w < 0.0);
        assert!(
            FemPhysicalPowerEvidence::from_operator_work(&operator_work, 0.0, 1.2)
                .unwrap_err()
                .contains("frequency")
        );
        assert!(
            FemPhysicalPowerEvidence::from_operator_work(&operator_work, 100.0, f64::NAN)
                .unwrap_err()
                .contains("density")
        );
    }

    #[test]
    fn boundary_flux_rejects_invalid_inputs() {
        let assembler = scalar_assembler();
        let mut boundary_coeffs = HashMap::new();
        boundary_coeffs.insert(7, Complex64::new(0.0, -0.3));
        let error = fem_boundary_flux_evidence(
            &assembler,
            &boundary_coeffs,
            0.0,
            2.0,
            &[Complex64::new(1.0, 0.0)],
        )
        .unwrap_err();
        assert!(error.contains("frequency"));

        let error =
            fem_boundary_flux_evidence(&assembler, &boundary_coeffs, 10.0, 2.0, &[]).unwrap_err();
        assert!(error.contains("dimensions"));

        let mut unknown_boundary = HashMap::new();
        unknown_boundary.insert(99, Complex64::new(0.0, -0.3));
        let error = fem_boundary_flux_evidence(
            &assembler,
            &unknown_boundary,
            10.0,
            2.0,
            &[Complex64::new(1.0, 0.0)],
        )
        .unwrap_err();
        assert!(error.contains("boundary"));
    }
}

impl EnergyBalanceReport {
    /// Require finite balance errors within relative and absolute budgets.
    pub fn require_within(
        &self,
        max_rms_relative_error: f64,
        max_max_relative_error: f64,
        max_absolute_error_w: f64,
    ) -> Result<(), String> {
        if !max_rms_relative_error.is_finite()
            || max_rms_relative_error < 0.0
            || !max_max_relative_error.is_finite()
            || max_max_relative_error < 0.0
            || !max_absolute_error_w.is_finite()
            || max_absolute_error_w < 0.0
        {
            return Err("energy-balance error budgets must be finite non-negative values".into());
        }
        if self.sample_count == 0 {
            return Err("energy-balance report requires at least one sample".into());
        }
        if !self.rms_relative_error.is_finite()
            || self.rms_relative_error < 0.0
            || !self.max_relative_error.is_finite()
            || self.max_relative_error < 0.0
            || !self.max_absolute_error_w.is_finite()
            || self.max_absolute_error_w < 0.0
        {
            return Err("energy-balance report contains invalid error metrics".into());
        }
        if self.rms_relative_error > max_rms_relative_error
            || self.max_relative_error > max_max_relative_error
            || self.max_absolute_error_w > max_absolute_error_w
        {
            return Err(format!(
                "energy-balance error exceeds budget: RMS={} (limit {}), max={} (limit {}), absolute={} W (limit {} W)",
                self.rms_relative_error,
                max_rms_relative_error,
                self.max_relative_error,
                max_max_relative_error,
                self.max_absolute_error_w,
                max_absolute_error_w,
            ));
        }
        Ok(())
    }
}

/// Check passive active-power balance over an ordered frequency grid.
///
/// The balance convention is
/// `source_power = absorbed_power + radiated_power`.  Negative source,
/// absorbed, or radiated powers are rejected as passivity violations rather
/// than hidden by an absolute-value operation.
pub fn check_acoustic_energy_balance(
    samples: &[AcousticPowerBalanceSample],
    power_floor_w: f64,
) -> Result<EnergyBalanceReport, String> {
    if !power_floor_w.is_finite() || power_floor_w <= 0.0 {
        return Err("energy-balance power floor must be finite positive".into());
    }
    if samples.is_empty() {
        return Err("energy-balance requires at least one frequency sample".into());
    }

    let mut sum_squared = 0.0f64;
    let mut max_relative_error = 0.0f64;
    let mut max_absolute_error_w = 0.0f64;
    let mut previous_frequency = None;
    for (index, sample) in samples.iter().enumerate() {
        if !sample.frequency_hz.is_finite() || sample.frequency_hz <= 0.0 {
            return Err(format!(
                "energy-balance frequency {index} must be finite positive"
            ));
        }
        if let Some(previous) = previous_frequency
            && sample.frequency_hz <= previous
        {
            return Err(format!(
                "energy-balance frequencies must be strictly increasing at index {index}"
            ));
        }
        previous_frequency = Some(sample.frequency_hz);
        if !sample.source_power_w.is_finite()
            || !sample.absorbed_power_w.is_finite()
            || !sample.radiated_power_w.is_finite()
        {
            return Err(format!(
                "energy-balance power row {index} contains a non-finite value"
            ));
        }
        if sample.source_power_w < 0.0
            || sample.absorbed_power_w < 0.0
            || sample.radiated_power_w < 0.0
        {
            return Err(format!(
                "energy-balance power row {index} violates passive non-negative-power contract"
            ));
        }
        let absolute_error_w =
            sample.source_power_w - sample.absorbed_power_w - sample.radiated_power_w;
        if !absolute_error_w.is_finite() {
            return Err(format!("energy-balance row {index} residual is non-finite"));
        }
        let relative_error = absolute_error_w.abs() / sample.source_power_w.max(power_floor_w);
        if !relative_error.is_finite() {
            return Err(format!(
                "energy-balance row {index} relative residual is non-finite"
            ));
        }
        sum_squared += relative_error * relative_error;
        if !sum_squared.is_finite() {
            return Err("energy-balance RMS accumulation is non-finite".into());
        }
        max_relative_error = max_relative_error.max(relative_error);
        max_absolute_error_w = max_absolute_error_w.max(absolute_error_w.abs());
    }
    let rms_relative_error = (sum_squared / samples.len() as f64).sqrt();
    if !rms_relative_error.is_finite()
        || !max_relative_error.is_finite()
        || !max_absolute_error_w.is_finite()
    {
        return Err("energy-balance report is non-finite".into());
    }
    Ok(EnergyBalanceReport {
        sample_count: samples.len(),
        rms_relative_error,
        max_relative_error,
        max_absolute_error_w,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_samples() -> Vec<AcousticPowerBalanceSample> {
        vec![
            AcousticPowerBalanceSample {
                frequency_hz: 100.0,
                source_power_w: 1.0,
                absorbed_power_w: 0.25,
                radiated_power_w: 0.75,
            },
            AcousticPowerBalanceSample {
                frequency_hz: 200.0,
                source_power_w: 2.0,
                absorbed_power_w: 0.5,
                radiated_power_w: 1.5,
            },
            AcousticPowerBalanceSample {
                frequency_hz: 400.0,
                source_power_w: 4.0,
                absorbed_power_w: 1.0,
                radiated_power_w: 3.0,
            },
        ]
    }

    #[test]
    fn exact_passive_balance_passes_relative_and_absolute_budgets() {
        let report = check_acoustic_energy_balance(&exact_samples(), 1.0e-12).unwrap();
        assert_eq!(report.sample_count, 3);
        assert_eq!(report.rms_relative_error, 0.0);
        assert_eq!(report.max_relative_error, 0.0);
        assert_eq!(report.max_absolute_error_w, 0.0);
        report.require_within(0.0, 0.0, 0.0).unwrap();
    }

    #[test]
    fn residual_is_reported_and_budgeted() {
        let mut samples = exact_samples();
        samples[1].radiated_power_w += 0.02;
        let report = check_acoustic_energy_balance(&samples, 1.0e-12).unwrap();
        assert!((report.max_absolute_error_w - 0.02).abs() < 1.0e-12);
        assert!(report.max_relative_error > 0.009);
        assert!(
            report
                .require_within(0.001, 0.001, 0.01)
                .unwrap_err()
                .contains("exceeds budget")
        );
        report.require_within(0.02, 0.02, 0.021).unwrap();
    }

    #[test]
    fn negative_power_is_rejected_as_non_passive() {
        let mut samples = exact_samples();
        samples[0].absorbed_power_w = -1.0e-6;
        let error = check_acoustic_energy_balance(&samples, 1.0e-12).unwrap_err();
        assert!(error.contains("non-negative-power"));
    }

    #[test]
    fn malformed_frequency_grid_is_rejected() {
        let mut samples = exact_samples();
        samples[1].frequency_hz = samples[0].frequency_hz;
        let error = check_acoustic_energy_balance(&samples, 1.0e-12).unwrap_err();
        assert!(error.contains("strictly increasing"));

        let mut samples = exact_samples();
        samples[0].frequency_hz = f64::NAN;
        let error = check_acoustic_energy_balance(&samples, 1.0e-12).unwrap_err();
        assert!(error.contains("finite positive"));
    }

    #[test]
    fn non_finite_power_and_residual_are_rejected() {
        let mut samples = exact_samples();
        samples[0].source_power_w = f64::NAN;
        let error = check_acoustic_energy_balance(&samples, 1.0e-12).unwrap_err();
        assert!(error.contains("non-finite value"));

        let mut samples = exact_samples();
        samples[0].source_power_w = f64::MAX;
        samples[0].absorbed_power_w = -f64::MIN_POSITIVE;
        let error = check_acoustic_energy_balance(&samples, 1.0e-12).unwrap_err();
        assert!(error.contains("non-negative-power"));
    }

    #[test]
    fn empty_and_invalid_floor_are_rejected() {
        assert!(
            check_acoustic_energy_balance(&[], 1.0e-12)
                .unwrap_err()
                .contains("at least one")
        );
        assert!(
            check_acoustic_energy_balance(&exact_samples(), 0.0)
                .unwrap_err()
                .contains("power floor")
        );
        assert!(
            check_acoustic_energy_balance(&exact_samples(), f64::NAN)
                .unwrap_err()
                .contains("power floor")
        );
    }

    #[test]
    fn invalid_report_metrics_and_budgets_are_rejected() {
        let report = check_acoustic_energy_balance(&exact_samples(), 1.0e-12).unwrap();
        assert!(
            report
                .require_within(f64::NAN, 0.1, 0.1)
                .unwrap_err()
                .contains("budgets")
        );
        let invalid = EnergyBalanceReport {
            sample_count: 0,
            rms_relative_error: 0.0,
            max_relative_error: 0.0,
            max_absolute_error_w: 0.0,
        };
        assert!(
            invalid
                .require_within(0.1, 0.1, 0.1)
                .unwrap_err()
                .contains("at least one")
        );
        let invalid = EnergyBalanceReport {
            sample_count: 1,
            rms_relative_error: f64::NAN,
            max_relative_error: 0.0,
            max_absolute_error_w: 0.0,
        };
        assert!(
            invalid
                .require_within(0.1, 0.1, 0.1)
                .unwrap_err()
                .contains("invalid error metrics")
        );
    }
}
