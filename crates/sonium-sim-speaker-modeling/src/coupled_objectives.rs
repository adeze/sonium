//! Shared driver/product metrics and constraint plumbing.
//!
//! The modeling crate owns the driver-unit quantities that product-side
//! optimizers consume.  Keeping these metrics in one typed record prevents a
//! sweep from reporting impedance, structural, nonlinear, and manufacturing
//! evidence through unrelated ad-hoc fields.  The module deliberately does
//! not run a solver: callers may populate the record from an axisymmetric
//! sweep, a resolved coupled solve, a large-signal result, or a calibrated
//! measurement.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sonium_solvers_optimization::{Constraint, ConstraintKind};

use crate::axisymmetric_linear::AxisymmetricSweepQa;
use crate::large_signal::{LargeSignalResult, ThermalCompressionResult};
use crate::product_design::ProductReferenceSweep;
use crate::structural_acoustic::StructuralAcousticResponse;
use crate::structural_constraints::{
    CabinetPanelEvaluation, MISSING_METRIC_MARGIN, StructuralEvaluation,
    moving_assembly_region_name,
};
use crate::{ModelingError, require_positive};

/// Scalar metrics shared by driver modeling and product optimization.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoupledProductMetrics {
    /// CEA-2034 preference score. Higher values are preferred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cea_preference_score: Option<f64>,
    /// Minimum real part of the electrical input impedance over the sweep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_impedance_real_ohm: Option<f64>,
    /// Maximum relative residual retained by the distributed FEM–BEM
    /// reference sweep.  This is optional evidence, but an active limit
    /// fails closed when the evidence is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_reference_relative_residual: Option<f64>,
    /// Maximum reduced-order residual checked during ROM promotion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_rom_reduced_residual: Option<f64>,
    /// Minimum active-subspace agreement retained by ROM validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_rom_rank_agreement: Option<f64>,
    /// Maximum relative gradient disagreement retained by ROM validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_rom_gradient_relative_error: Option<f64>,
    /// Maximum product-constraint disagreement retained by ROM validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_rom_constraint_violation: Option<f64>,
    /// Maximum one-way peak moving-assembly excursion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_excursion_m: Option<f64>,
    /// Maximum peak stress by moving-assembly material marker.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub peak_stress_pa: BTreeMap<i32, f64>,
    /// Maximum peak center displacement among cabinet panels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_panel_displacement_m: Option<f64>,
    /// Maximum peak center bending stress among cabinet panels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_panel_bending_stress_pa: Option<f64>,
    /// Maximum displacement THD as a fraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_distortion_fraction: Option<f64>,
    /// Maximum positive thermal compression in dB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_compression_db: Option<f64>,
    /// Estimated product cost in the caller's configured currency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Non-negative manufacturability penalty; zero is the preferred design.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manufacturability_penalty: Option<f64>,
}

impl CoupledProductMetrics {
    /// Build the linear and structural portion of the contract from one
    /// axisymmetric sweep.  The source values are copied, so later reporting
    /// cannot accidentally observe a mutable solver QA accumulator.
    pub fn from_axisymmetric_sweep(
        sweep_qa: &AxisymmetricSweepQa,
        structural: &StructuralEvaluation,
    ) -> Self {
        let maximum_panel_displacement_m = structural
            .panel_peak_displacement_m
            .iter()
            .copied()
            .reduce(f64::max);
        let maximum_panel_bending_stress_pa = structural
            .panel_peak_bending_stress_pa
            .iter()
            .copied()
            .reduce(f64::max);
        Self {
            minimum_impedance_real_ohm: Some(sweep_qa.minimum_impedance_real_ohm),
            peak_excursion_m: structural.peak_excursion_m,
            peak_stress_pa: structural.peak_stress_pa.clone(),
            maximum_panel_displacement_m,
            maximum_panel_bending_stress_pa,
            ..Self::default()
        }
    }

    /// Add nonlinear displacement and distortion evidence to an existing
    /// linear/structural record.
    pub fn with_large_signal(mut self, result: &LargeSignalResult) -> Self {
        self.peak_excursion_m = Some(
            self.peak_excursion_m
                .unwrap_or(0.0)
                .max(result.peak_displacement_m),
        );
        self.maximum_distortion_fraction = Some(
            self.maximum_distortion_fraction
                .unwrap_or(0.0)
                .max(result.displacement_thd),
        );
        self
    }

    /// Add a CEA-2034 preference score to the product evidence.
    pub fn with_cea_preference_score(mut self, preference_score: f64) -> Self {
        self.cea_preference_score = Some(preference_score);
        self
    }

    /// Add residual evidence from a validated distributed FEM–BEM reference
    /// sweep.  The complete sweep remains attached to the product simulator
    /// result; metrics retain only the scalar used by acceptance constraints.
    pub fn with_reference_sweep(
        mut self,
        sweep: &ProductReferenceSweep,
    ) -> Result<Self, ModelingError> {
        sweep
            .validate()
            .map_err(|error| ModelingError::InvalidReferenceEvidence(error.to_string()))?;
        self.maximum_reference_relative_residual = sweep.maximum_relative_residual();
        if let Some(rom) = &sweep.rom_promotion {
            self.maximum_rom_reduced_residual = rom.maximum_checked_reduced_residual;
            self.minimum_rom_rank_agreement = rom.minimum_checked_rank_agreement;
            self.maximum_rom_gradient_relative_error = rom.maximum_checked_gradient_relative_error;
            self.maximum_rom_constraint_violation = rom.maximum_checked_constraint_violation;
        }
        Ok(self)
    }

    /// Add the worst linear cabinet-panel response derived from the retained
    /// product reference sweep.
    pub fn with_reference_panel_response(
        mut self,
        evaluations: &[CabinetPanelEvaluation],
    ) -> Result<Self, ModelingError> {
        if evaluations.is_empty() {
            return Err(ModelingError::OutOfRange {
                field: "coupled_product_metrics.reference_panel_response",
                constraint: "non-empty",
            });
        }
        if evaluations.iter().any(|evaluation| {
            !evaluation.worst_frequency_hz.is_finite()
                || evaluation.worst_frequency_hz <= 0.0
                || !evaluation.peak_displacement_m.is_finite()
                || evaluation.peak_displacement_m < 0.0
                || !evaluation.peak_bending_stress_pa.is_finite()
                || evaluation.peak_bending_stress_pa < 0.0
        }) {
            return Err(ModelingError::InvalidReferenceEvidence(
                "cabinet panel response must be finite and non-negative".to_owned(),
            ));
        }
        self.maximum_panel_displacement_m = evaluations
            .iter()
            .map(|evaluation| evaluation.peak_displacement_m)
            .reduce(f64::max);
        self.maximum_panel_bending_stress_pa = evaluations
            .iter()
            .map(|evaluation| evaluation.peak_bending_stress_pa)
            .reduce(f64::max);
        Ok(self)
    }

    /// Add frequency-domain structural displacement/stress evidence retained
    /// by the product BEM.  Structural responses are RMS modeling phasors;
    /// product limits use peak amplitudes, so the conversion is explicit here
    /// at the metrics boundary.  An all-`None` vector represents a product
    /// without the optional structural model and leaves the metrics absent;
    /// a partially populated vector is malformed evidence and is rejected.
    pub fn with_structural_responses(
        mut self,
        frequencies_hz: &[f64],
        responses: &[Option<StructuralAcousticResponse>],
    ) -> Result<Self, ModelingError> {
        if responses.is_empty() {
            return Ok(self);
        }
        if responses.len() != frequencies_hz.len() {
            return Err(ModelingError::InvalidReferenceEvidence(format!(
                "structural response length {} does not match frequency count {}",
                responses.len(),
                frequencies_hz.len()
            )));
        }
        let present = responses
            .iter()
            .filter_map(Option::as_ref)
            .collect::<Vec<_>>();
        if present.is_empty() {
            return Ok(self);
        }
        if present.len() != responses.len() {
            return Err(ModelingError::InvalidReferenceEvidence(
                "structural responses must be present at every frequency when enabled".to_owned(),
            ));
        }
        let mut maximum_displacement_m: f64 = 0.0;
        let mut maximum_stress_pa: f64 = 0.0;
        for (index, (frequency_hz, response)) in frequencies_hz.iter().zip(present).enumerate() {
            if !frequency_hz.is_finite() || *frequency_hz <= 0.0 {
                return Err(ModelingError::InvalidReferenceEvidence(format!(
                    "structural frequency at index {index} must be finite and positive"
                )));
            }
            let scale = frequency_hz.abs().max(response.frequency_hz.abs()).max(1.0);
            if !response.frequency_hz.is_finite()
                || (response.frequency_hz - frequency_hz).abs() > 1.0e-9 * scale
            {
                return Err(ModelingError::InvalidReferenceEvidence(format!(
                    "structural frequency at index {index} does not match simulation frequency"
                )));
            }
            for (field, value) in [
                ("displacement_rms_m", response.displacement_rms_m),
                ("velocity_rms_m_per_s", response.velocity_rms_m_per_s),
                ("stress_rms_pa", response.stress_rms_pa),
                (
                    "volume_velocity_rms_m3_per_s",
                    response.volume_velocity_rms_m3_per_s,
                ),
            ] {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(ModelingError::NonFinite { field });
                }
            }
            if !response.absorbed_power_w.is_finite() || response.absorbed_power_w < -1.0e-10 {
                return Err(ModelingError::InvalidRadiationImpedance);
            }
            maximum_displacement_m = maximum_displacement_m
                .max(response.displacement_rms_m.norm() * std::f64::consts::SQRT_2);
            maximum_stress_pa =
                maximum_stress_pa.max(response.stress_rms_pa.norm() * std::f64::consts::SQRT_2);
        }
        self.maximum_panel_displacement_m = Some(
            self.maximum_panel_displacement_m
                .unwrap_or(0.0)
                .max(maximum_displacement_m),
        );
        self.maximum_panel_bending_stress_pa = Some(
            self.maximum_panel_bending_stress_pa
                .unwrap_or(0.0)
                .max(maximum_stress_pa),
        );
        Ok(self)
    }

    /// Add thermal compression evidence, retaining the worst point when a
    /// caller combines several frequencies or drive levels.
    pub fn with_thermal_compression(mut self, result: &ThermalCompressionResult) -> Self {
        self.thermal_compression_db = Some(
            self.thermal_compression_db
                .unwrap_or(0.0)
                .max(result.compression_db.max(0.0)),
        );
        self
    }

    pub fn with_distortion_fraction(mut self, distortion: f64) -> Self {
        self.maximum_distortion_fraction = Some(
            self.maximum_distortion_fraction
                .unwrap_or(0.0)
                .max(distortion),
        );
        self
    }

    pub fn with_peak_excursion_m(mut self, excursion_m: f64) -> Self {
        self.peak_excursion_m = Some(self.peak_excursion_m.unwrap_or(0.0).max(excursion_m));
        self
    }

    pub fn with_thermal_compression_db(mut self, compression_db: f64) -> Self {
        self.thermal_compression_db = Some(
            self.thermal_compression_db
                .unwrap_or(0.0)
                .max(compression_db.max(0.0)),
        );
        self
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    pub fn with_manufacturability_penalty(mut self, penalty: f64) -> Self {
        self.manufacturability_penalty = Some(penalty);
        self
    }

    pub fn validate(&self) -> Result<(), ModelingError> {
        validate_optional_finite(
            "coupled_product_metrics.cea_preference_score",
            self.cea_preference_score,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.minimum_impedance_real_ohm",
            self.minimum_impedance_real_ohm,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.maximum_reference_relative_residual",
            self.maximum_reference_relative_residual,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.maximum_rom_reduced_residual",
            self.maximum_rom_reduced_residual,
        )?;
        if let Some(value) = self.minimum_rom_rank_agreement
            && (!value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(ModelingError::OutOfRange {
                field: "coupled_product_metrics.minimum_rom_rank_agreement",
                constraint: "finite in [0, 1]",
            });
        }
        validate_optional_nonnegative(
            "coupled_product_metrics.maximum_rom_gradient_relative_error",
            self.maximum_rom_gradient_relative_error,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.maximum_rom_constraint_violation",
            self.maximum_rom_constraint_violation,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.peak_excursion_m",
            self.peak_excursion_m,
        )?;
        for value in self.peak_stress_pa.values() {
            validate_nonnegative("coupled_product_metrics.peak_stress_pa", *value)?;
        }
        validate_optional_nonnegative(
            "coupled_product_metrics.maximum_panel_displacement_m",
            self.maximum_panel_displacement_m,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.maximum_panel_bending_stress_pa",
            self.maximum_panel_bending_stress_pa,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.maximum_distortion_fraction",
            self.maximum_distortion_fraction,
        )?;
        validate_optional_nonnegative(
            "coupled_product_metrics.thermal_compression_db",
            self.thermal_compression_db,
        )?;
        validate_optional_nonnegative("coupled_product_metrics.cost", self.cost)?;
        validate_optional_nonnegative(
            "coupled_product_metrics.manufacturability_penalty",
            self.manufacturability_penalty,
        )?;
        Ok(())
    }
}

/// Product limits for the coupled driver state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoupledProductLimits {
    /// Minimum acceptable CEA-2034 preference score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_cea_preference_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_impedance_real_ohm: Option<f64>,
    /// Maximum acceptable residual of the distributed FEM–BEM reference
    /// evidence. Missing reference evidence is infeasible when this is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_reference_relative_residual: Option<f64>,
    /// Maximum accepted reduced-order residual. Missing ROM evidence fails
    /// closed when any ROM limit is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_rom_reduced_residual: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_rom_rank_agreement: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_rom_gradient_relative_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_rom_constraint_violation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_excursion_m: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub maximum_stress_pa: BTreeMap<i32, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_panel_displacement_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_panel_bending_stress_pa: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_distortion_fraction: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_thermal_compression_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_manufacturability_penalty: Option<f64>,
}

impl CoupledProductLimits {
    pub fn validate(&self) -> Result<(), ModelingError> {
        validate_optional_finite(
            "coupled_product_limits.minimum_cea_preference_score",
            self.minimum_cea_preference_score,
        )?;
        validate_optional_positive(
            "coupled_product_limits.minimum_impedance_real_ohm",
            self.minimum_impedance_real_ohm,
        )?;
        validate_optional_nonnegative(
            "coupled_product_limits.maximum_reference_relative_residual",
            self.maximum_reference_relative_residual,
        )?;
        for (field, value) in [
            (
                "coupled_product_limits.maximum_rom_reduced_residual",
                self.maximum_rom_reduced_residual,
            ),
            (
                "coupled_product_limits.maximum_rom_gradient_relative_error",
                self.maximum_rom_gradient_relative_error,
            ),
            (
                "coupled_product_limits.maximum_rom_constraint_violation",
                self.maximum_rom_constraint_violation,
            ),
        ] {
            validate_optional_nonnegative(field, value)?;
        }
        if let Some(value) = self.minimum_rom_rank_agreement
            && (!value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(ModelingError::OutOfRange {
                field: "coupled_product_limits.minimum_rom_rank_agreement",
                constraint: "finite in [0, 1]",
            });
        }
        validate_optional_positive(
            "coupled_product_limits.maximum_excursion_m",
            self.maximum_excursion_m,
        )?;
        for value in self.maximum_stress_pa.values() {
            require_positive("coupled_product_limits.maximum_stress_pa", *value)?;
        }
        for (field, value) in [
            (
                "coupled_product_limits.maximum_panel_displacement_m",
                self.maximum_panel_displacement_m,
            ),
            (
                "coupled_product_limits.maximum_panel_bending_stress_pa",
                self.maximum_panel_bending_stress_pa,
            ),
            (
                "coupled_product_limits.maximum_distortion_fraction",
                self.maximum_distortion_fraction,
            ),
            (
                "coupled_product_limits.maximum_thermal_compression_db",
                self.maximum_thermal_compression_db,
            ),
            ("coupled_product_limits.maximum_cost", self.maximum_cost),
            (
                "coupled_product_limits.maximum_manufacturability_penalty",
                self.maximum_manufacturability_penalty,
            ),
        ] {
            validate_optional_nonnegative(field, value)?;
        }
        Ok(())
    }
}

/// Labels for the ordered inequality vector returned by
/// [`coupled_product_constraint_margins`].
pub fn coupled_product_constraint_labels(limits: &CoupledProductLimits) -> Vec<String> {
    let mut labels = Vec::new();
    if limits.minimum_cea_preference_score.is_some() {
        labels.push("cea2034.minimum_preference_score".to_owned());
    }
    if limits.minimum_impedance_real_ohm.is_some() {
        labels.push("electrical.minimum_impedance_real".to_owned());
    }
    if limits.maximum_reference_relative_residual.is_some() {
        labels.push("coupling.maximum_reference_relative_residual".to_owned());
    }
    if limits.maximum_rom_reduced_residual.is_some() {
        labels.push("coupling.maximum_rom_reduced_residual".to_owned());
    }
    if limits.minimum_rom_rank_agreement.is_some() {
        labels.push("coupling.minimum_rom_rank_agreement".to_owned());
    }
    if limits.maximum_rom_gradient_relative_error.is_some() {
        labels.push("coupling.maximum_rom_gradient_relative_error".to_owned());
    }
    if limits.maximum_rom_constraint_violation.is_some() {
        labels.push("coupling.maximum_rom_constraint_violation".to_owned());
    }
    if limits.maximum_excursion_m.is_some() {
        labels.push("moving_assembly.maximum_excursion".to_owned());
    }
    for marker in limits.maximum_stress_pa.keys() {
        let region = moving_assembly_region_name(*marker).unwrap_or("unknown");
        labels.push(format!("moving_assembly.stress[{region}]"));
    }
    if limits.maximum_panel_displacement_m.is_some() {
        labels.push("cabinet.maximum_panel_displacement".to_owned());
    }
    if limits.maximum_panel_bending_stress_pa.is_some() {
        labels.push("cabinet.maximum_panel_bending_stress".to_owned());
    }
    if limits.maximum_distortion_fraction.is_some() {
        labels.push("nonlinear.maximum_distortion".to_owned());
    }
    if limits.maximum_thermal_compression_db.is_some() {
        labels.push("thermal.maximum_compression_db".to_owned());
    }
    if limits.maximum_cost.is_some() {
        labels.push("manufacturing.maximum_cost".to_owned());
    }
    if limits.maximum_manufacturability_penalty.is_some() {
        labels.push("manufacturing.maximum_penalty".to_owned());
    }
    labels
}

/// Evaluate all active coupled product constraints as `limit - value`.
/// Feasible margins are non-negative. Missing metrics are fail-safe and use
/// [`MISSING_METRIC_MARGIN`], matching the existing structural contract.
pub fn coupled_product_constraint_margins(
    metrics: &CoupledProductMetrics,
    limits: &CoupledProductLimits,
) -> Vec<f64> {
    let mut margins = Vec::new();
    if let Some(limit) = limits.minimum_cea_preference_score {
        margins.push(lower_bound_margin(metrics.cea_preference_score, limit));
    }
    if let Some(limit) = limits.minimum_impedance_real_ohm {
        margins.push(optional_margin(metrics.minimum_impedance_real_ohm, limit));
    }
    if let Some(limit) = limits.maximum_reference_relative_residual {
        margins.push(upper_bound_margin(
            metrics.maximum_reference_relative_residual,
            limit,
        ));
    }
    if let Some(limit) = limits.maximum_rom_reduced_residual {
        margins.push(upper_bound_margin(
            metrics.maximum_rom_reduced_residual,
            limit,
        ));
    }
    if let Some(limit) = limits.minimum_rom_rank_agreement {
        margins.push(lower_bound_margin(
            metrics.minimum_rom_rank_agreement,
            limit,
        ));
    }
    if let Some(limit) = limits.maximum_rom_gradient_relative_error {
        margins.push(upper_bound_margin(
            metrics.maximum_rom_gradient_relative_error,
            limit,
        ));
    }
    if let Some(limit) = limits.maximum_rom_constraint_violation {
        margins.push(upper_bound_margin(
            metrics.maximum_rom_constraint_violation,
            limit,
        ));
    }
    if let Some(limit) = limits.maximum_excursion_m {
        margins.push(upper_bound_margin(metrics.peak_excursion_m, limit));
    }
    for (marker, limit) in &limits.maximum_stress_pa {
        margins.push(
            metrics
                .peak_stress_pa
                .get(marker)
                .map_or(MISSING_METRIC_MARGIN, |value| limit - value),
        );
    }
    if let Some(limit) = limits.maximum_panel_displacement_m {
        margins.push(upper_bound_margin(
            metrics.maximum_panel_displacement_m,
            limit,
        ));
    }
    if let Some(limit) = limits.maximum_panel_bending_stress_pa {
        margins.push(upper_bound_margin(
            metrics.maximum_panel_bending_stress_pa,
            limit,
        ));
    }
    if let Some(limit) = limits.maximum_distortion_fraction {
        margins.push(upper_bound_margin(
            metrics.maximum_distortion_fraction,
            limit,
        ));
    }
    if let Some(limit) = limits.maximum_thermal_compression_db {
        margins.push(upper_bound_margin(metrics.thermal_compression_db, limit));
    }
    if let Some(limit) = limits.maximum_cost {
        margins.push(upper_bound_margin(metrics.cost, limit));
    }
    if let Some(limit) = limits.maximum_manufacturability_penalty {
        margins.push(upper_bound_margin(metrics.manufacturability_penalty, limit));
    }
    margins
}

/// Report and feasibility summary for an evaluated coupled product state.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CoupledProductConstraintReport {
    pub labels: Vec<String>,
    pub margins: Vec<f64>,
    pub feasible: bool,
    pub minimum_margin: Option<f64>,
}

impl CoupledProductConstraintReport {
    pub fn evaluate(
        metrics: &CoupledProductMetrics,
        limits: &CoupledProductLimits,
    ) -> Result<Self, ModelingError> {
        metrics.validate()?;
        limits.validate()?;
        if limits.maximum_excursion_m.is_some() && metrics.peak_excursion_m.is_none() {
            return Err(ModelingError::InvalidReferenceEvidence(
                "moving-assembly excursion evidence is required when an excursion limit is configured"
                    .to_owned(),
            ));
        }
        let labels = coupled_product_constraint_labels(limits);
        let margins = coupled_product_constraint_margins(metrics, limits);
        let minimum_margin = margins.iter().copied().reduce(f64::min);
        Ok(Self {
            labels,
            feasible: margins.iter().all(|margin| *margin >= 0.0),
            margins,
            minimum_margin,
        })
    }
}

/// Bind the coupled product limits to a solver inequality constraint. The
/// evaluator is called exactly once for each constraint callback, and an
/// invalid/missing evaluation becomes an infeasible vector rather than an
/// accidental feasible point.
pub fn to_coupled_product_constraints<F>(
    limits: &CoupledProductLimits,
    evaluate: F,
) -> CoupledProductConstraintSet
where
    F: Fn(&[f64]) -> CoupledProductMetrics + Send + Sync + 'static,
{
    let labels = coupled_product_constraint_labels(limits);
    if labels.is_empty() {
        return CoupledProductConstraintSet {
            constraints: Vec::new(),
            labels,
        };
    }
    let limits = limits.clone();
    let dim = labels.len();
    let constraint = Constraint {
        kind: ConstraintKind::Inequality,
        dim,
        fun: Box::new(move |x: &[f64]| {
            let metrics = evaluate(x);
            if metrics.validate().is_err() || limits.validate().is_err() {
                vec![MISSING_METRIC_MARGIN; dim]
            } else {
                coupled_product_constraint_margins(&metrics, &limits)
            }
        }),
        jac: None,
    };
    CoupledProductConstraintSet {
        constraints: vec![constraint],
        labels,
    }
}

/// Solver-ready coupled product constraints plus reporting labels.
pub struct CoupledProductConstraintSet {
    pub constraints: Vec<Constraint<f64>>,
    pub labels: Vec<String>,
}

fn optional_margin(value: Option<f64>, limit: f64) -> f64 {
    value.map_or(MISSING_METRIC_MARGIN, |value| limit - value)
}

fn upper_bound_margin(value: Option<f64>, limit: f64) -> f64 {
    optional_margin(value, limit)
}

fn lower_bound_margin(value: Option<f64>, limit: f64) -> f64 {
    value.map_or(MISSING_METRIC_MARGIN, |value| value - limit)
}

fn validate_optional_positive(
    field: &'static str,
    value: Option<f64>,
) -> Result<(), ModelingError> {
    if let Some(value) = value {
        require_positive(field, value)?;
    }
    Ok(())
}

fn validate_optional_nonnegative(
    field: &'static str,
    value: Option<f64>,
) -> Result<(), ModelingError> {
    if let Some(value) = value {
        validate_nonnegative(field, value)?;
    }
    Ok(())
}

fn validate_optional_finite(field: &'static str, value: Option<f64>) -> Result<(), ModelingError> {
    if let Some(value) = value
        && !value.is_finite()
    {
        return Err(ModelingError::NonFinite { field });
    }
    Ok(())
}

fn validate_nonnegative(field: &'static str, value: f64) -> Result<(), ModelingError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ModelingError::OutOfRange {
            field,
            constraint: "finite and non-negative",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axisymmetric_linear::AxisymmetricSweepQa;
    use crate::structural_constraints::StructuralEvaluation;
    use num_complex::Complex64;

    #[test]
    fn metrics_compose_linear_structural_nonlinear_and_thermal_evidence() {
        let metrics = CoupledProductMetrics::from_axisymmetric_sweep(
            &AxisymmetricSweepQa {
                max_relative_linear_residual: 1.0e-9,
                max_relative_power_residual: 2.0e-9,
                max_reciprocity_defect: 3.0e-9,
                minimum_impedance_real_ohm: 3.2,
                minimum_dissipated_power_w: 0.1,
                all_points_converged: true,
            },
            &StructuralEvaluation {
                peak_excursion_m: Some(0.001),
                peak_stress_pa: [(7, 2.0e6)].into_iter().collect(),
                panel_peak_displacement_m: vec![0.0002, 0.0001],
                panel_peak_bending_stress_pa: vec![4.0e6, 3.0e6],
            },
        )
        .with_peak_excursion_m(0.002)
        .with_distortion_fraction(0.03)
        .with_thermal_compression_db(2.5);
        assert_eq!(metrics.peak_excursion_m, Some(0.002));
        assert_eq!(metrics.maximum_panel_displacement_m, Some(0.0002));
        assert_eq!(metrics.maximum_distortion_fraction, Some(0.03));
        assert_eq!(metrics.thermal_compression_db, Some(2.5));
        metrics.validate().unwrap();
    }

    #[test]
    fn constraint_report_is_ordered_fail_safe_and_solver_bindable() {
        let metrics = CoupledProductMetrics {
            minimum_impedance_real_ohm: Some(2.0),
            peak_excursion_m: Some(0.003),
            ..CoupledProductMetrics::default()
        };
        let limits = CoupledProductLimits {
            minimum_impedance_real_ohm: Some(3.0),
            maximum_excursion_m: Some(0.002),
            maximum_cost: Some(10.0),
            ..CoupledProductLimits::default()
        };
        let report = CoupledProductConstraintReport::evaluate(&metrics, &limits).unwrap();
        assert_eq!(report.labels.len(), 3);
        assert_eq!(report.margins, vec![1.0, -0.001, MISSING_METRIC_MARGIN]);
        assert!(!report.feasible);

        let set = to_coupled_product_constraints(&limits, move |_| metrics.clone());
        assert_eq!(set.labels, report.labels);
        let values = (set.constraints[0].fun)(&[]);
        assert_eq!(values, report.margins);
    }

    #[test]
    fn optional_limits_validate_physical_signs() {
        let limits = CoupledProductLimits {
            minimum_impedance_real_ohm: Some(0.0),
            ..CoupledProductLimits::default()
        };
        assert!(limits.validate().is_err());
        let metrics = CoupledProductMetrics {
            cost: Some(-1.0),
            ..CoupledProductMetrics::default()
        };
        assert!(metrics.validate().is_err());
    }

    #[test]
    fn reference_residual_limit_is_fail_safe_and_explainable() {
        let limits = CoupledProductLimits {
            maximum_reference_relative_residual: Some(1.0e-6),
            ..CoupledProductLimits::default()
        };
        let missing =
            CoupledProductConstraintReport::evaluate(&CoupledProductMetrics::default(), &limits)
                .unwrap();
        assert_eq!(
            missing.labels,
            vec!["coupling.maximum_reference_relative_residual"]
        );
        assert_eq!(missing.margins, vec![MISSING_METRIC_MARGIN]);
        assert!(!missing.feasible);

        let passing = CoupledProductMetrics {
            maximum_reference_relative_residual: Some(2.0e-7),
            ..CoupledProductMetrics::default()
        };
        let report = CoupledProductConstraintReport::evaluate(&passing, &limits).unwrap();
        assert_eq!(report.margins, vec![8.0e-7]);
        assert!(report.feasible);
    }

    #[test]
    fn rom_promotion_limits_are_fail_safe_and_explainable() {
        let limits = CoupledProductLimits {
            maximum_rom_reduced_residual: Some(1.0e-2),
            minimum_rom_rank_agreement: Some(0.9),
            maximum_rom_gradient_relative_error: Some(0.05),
            maximum_rom_constraint_violation: Some(0.01),
            ..CoupledProductLimits::default()
        };
        let missing =
            CoupledProductConstraintReport::evaluate(&CoupledProductMetrics::default(), &limits)
                .unwrap();
        assert_eq!(missing.labels.len(), 4);
        assert!(missing.margins.iter().all(|margin| *margin < 0.0));
        assert!(!missing.feasible);

        let passing = CoupledProductMetrics {
            maximum_rom_reduced_residual: Some(0.005),
            minimum_rom_rank_agreement: Some(0.95),
            maximum_rom_gradient_relative_error: Some(0.01),
            maximum_rom_constraint_violation: Some(0.001),
            ..CoupledProductMetrics::default()
        };
        let report = CoupledProductConstraintReport::evaluate(&passing, &limits).unwrap();
        assert!(report.feasible);
    }

    #[test]
    fn structural_responses_become_peak_displacement_and_stress_metrics() {
        let responses = vec![
            Some(StructuralAcousticResponse {
                frequency_hz: 100.0,
                displacement_rms_m: Complex64::new(1.0e-3, 0.0),
                stress_rms_pa: Complex64::new(2.0e3, 0.0),
                absorbed_power_w: 0.2,
                ..StructuralAcousticResponse::default()
            }),
            Some(StructuralAcousticResponse {
                frequency_hz: 200.0,
                displacement_rms_m: Complex64::new(2.0e-3, 0.0),
                stress_rms_pa: Complex64::new(1.0e3, 0.0),
                absorbed_power_w: 0.3,
                ..StructuralAcousticResponse::default()
            }),
        ];
        let metrics = CoupledProductMetrics::default()
            .with_structural_responses(&[100.0, 200.0], &responses)
            .unwrap();
        assert_eq!(
            metrics.maximum_panel_displacement_m,
            Some(2.0e-3 * std::f64::consts::SQRT_2)
        );
        assert_eq!(
            metrics.maximum_panel_bending_stress_pa,
            Some(2.0e3 * std::f64::consts::SQRT_2)
        );
        let limits = CoupledProductLimits {
            maximum_panel_displacement_m: Some(3.0e-3),
            maximum_panel_bending_stress_pa: Some(3.0e3),
            ..CoupledProductLimits::default()
        };
        assert!(
            CoupledProductConstraintReport::evaluate(&metrics, &limits)
                .unwrap()
                .feasible
        );
    }

    #[test]
    fn partial_structural_response_evidence_is_rejected() {
        let responses = vec![
            Some(StructuralAcousticResponse {
                frequency_hz: 100.0,
                ..StructuralAcousticResponse::default()
            }),
            None,
        ];
        assert!(
            CoupledProductMetrics::default()
                .with_structural_responses(&[100.0, 200.0], &responses)
                .is_err()
        );
    }

    #[test]
    fn preference_score_uses_a_lower_bound_and_allows_negative_scores() {
        let metrics = CoupledProductMetrics::default().with_cea_preference_score(-1.5);
        let limits = CoupledProductLimits {
            minimum_cea_preference_score: Some(-2.0),
            ..CoupledProductLimits::default()
        };
        let report = CoupledProductConstraintReport::evaluate(&metrics, &limits).unwrap();
        assert_eq!(report.labels, vec!["cea2034.minimum_preference_score"]);
        assert_eq!(report.margins, vec![0.5]);
        assert!(report.feasible);
    }

    #[test]
    fn configured_excursion_limit_requires_measured_or_solved_evidence() {
        let limits = CoupledProductLimits {
            maximum_excursion_m: Some(1.0e-3),
            ..CoupledProductLimits::default()
        };
        let error =
            CoupledProductConstraintReport::evaluate(&CoupledProductMetrics::default(), &limits)
                .expect_err("missing excursion evidence must not be promoted");
        assert!(error.to_string().contains("excursion evidence"));

        let metrics = CoupledProductMetrics::default().with_peak_excursion_m(5.0e-4);
        assert!(
            CoupledProductConstraintReport::evaluate(&metrics, &limits)
                .expect("complete excursion evidence should validate")
                .feasible
        );
    }
}
