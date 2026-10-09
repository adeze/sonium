//! Checked reduced-order uncertainty/scenario screening for propeller operation.
//!
//! Scenarios perturb source loads and operating conditions, then reuse the
//! ideal actuator-disk multipoint envelope. This provides deterministic
//! worst-case margin evidence without claiming robust shape, wake, CFD, or
//! measured uncertainty propagation.

use std::fmt;

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

use crate::{
    PropellerMultipointOperatingPoint, PropellerOperatingCondition, PropellerOperatingEnvelope,
    PropellerOperatingEnvelopeError,
};

/// One deterministic source/operating-condition uncertainty scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct PropellerUncertaintyScenario {
    /// Human-readable stable scenario label.
    pub label: String,
    /// Multiplicative thrust scale applied to the source ledger.
    pub thrust_scale: f64,
    /// Multiplicative torque scale applied to the source ledger.
    pub torque_scale: f64,
    /// Multiplicative RPM scale applied to source and conditions.
    pub rpm_scale: f64,
    /// Multiplicative freestream scale applied to conditions.
    pub freestream_scale: f64,
    /// Multiplicative density scale applied to conditions.
    pub density_scale: f64,
    /// Additive inflow-angle offset [rad].
    pub inflow_angle_offset_rad: f64,
    /// Additive temperature offset [K].
    pub temperature_offset_k: f64,
}

impl PropellerUncertaintyScenario {
    /// Nominal zero-perturbation scenario.
    pub fn nominal() -> Self {
        Self {
            label: "nominal".to_owned(),
            thrust_scale: 1.0,
            torque_scale: 1.0,
            rpm_scale: 1.0,
            freestream_scale: 1.0,
            density_scale: 1.0,
            inflow_angle_offset_rad: 0.0,
            temperature_offset_k: 0.0,
        }
    }
}

/// Options for robust scenario screening.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustOperatingOptions {
    /// Scenarios evaluated in caller order.
    pub scenarios: Vec<PropellerUncertaintyScenario>,
    /// Nominal operating conditions perturbed per scenario.
    pub conditions: Vec<PropellerOperatingCondition>,
}

/// One scenario's retained multipoint results and admission status.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustOperatingScenarioResult {
    /// Scenario applied to this result.
    pub scenario: PropellerUncertaintyScenario,
    /// Admitted multipoint results, in condition order.
    pub operating_points: Vec<PropellerMultipointOperatingPoint>,
    /// Minimum power margin among admitted points [W], if any.
    pub minimum_power_margin_w: Option<f64>,
    /// Minimum propulsive efficiency among admitted points, if any.
    pub minimum_propulsive_efficiency: Option<f64>,
    /// Whether every configured condition was admitted.
    pub admissible: bool,
    /// First condition rejected by the operating-envelope gate, if any.
    pub rejected_condition_index: Option<usize>,
    /// Underlying rejection evidence for an inadmissible scenario.
    pub rejection: Option<PropellerOperatingEnvelopeError>,
}

impl RobustOperatingScenarioResult {
    fn is_finite(&self) -> bool {
        scenario_is_finite(&self.scenario)
            && self.operating_points.iter().all(operating_point_is_finite)
            && self
                .minimum_power_margin_w
                .map(f64::is_finite)
                .unwrap_or(true)
            && self
                .minimum_propulsive_efficiency
                .map(f64::is_finite)
                .unwrap_or(true)
            && self
                .rejected_condition_index
                .map(|index| index < self.operating_points.len() + 1)
                .unwrap_or(true)
    }
}

/// Worst-case result over all configured uncertainty scenarios.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustOperatingResult {
    /// Per-scenario evidence in caller order.
    pub scenarios: Vec<RobustOperatingScenarioResult>,
    /// Whether every scenario/condition passed the envelope admission gate.
    pub all_admissible: bool,
    /// Scenario index associated with the worst available margin/rejection.
    pub worst_case_scenario_index: usize,
    /// Condition index associated with the worst available margin/rejection.
    pub worst_case_condition_index: usize,
    /// Minimum power margin over admitted points, if any.
    pub worst_power_margin_w: Option<f64>,
    /// Minimum propulsive efficiency over admitted points, if any.
    pub worst_propulsive_efficiency: Option<f64>,
}

impl RobustOperatingResult {
    /// Whether the result is finite and structurally consistent.
    pub fn is_finite(&self) -> bool {
        !self.scenarios.is_empty()
            && self.worst_case_scenario_index < self.scenarios.len()
            && self
                .scenarios
                .iter()
                .all(RobustOperatingScenarioResult::is_finite)
            && self
                .worst_power_margin_w
                .map(f64::is_finite)
                .unwrap_or(true)
            && self
                .worst_propulsive_efficiency
                .map(f64::is_finite)
                .unwrap_or(true)
    }

    /// Whether all configured scenarios and conditions were admitted.
    pub fn is_feasible(&self) -> bool {
        self.is_finite() && self.all_admissible
    }
}

/// Failure-closed error from robust operating screening.
#[derive(Debug, Clone, PartialEq)]
pub enum RobustOperatingError {
    /// The scenario matrix configuration is invalid.
    InvalidConfiguration {
        /// Configuration field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A non-admission error occurred while constructing or evaluating a
    /// scenario.
    ScenarioEvaluation {
        /// Scenario index.
        scenario_index: usize,
        /// Condition index, when evaluation reached a condition.
        condition_index: Option<usize>,
        /// Underlying checked envelope error.
        error: PropellerOperatingEnvelopeError,
    },
    /// The aggregate result became non-finite.
    NonFiniteResult,
}

impl fmt::Display for RobustOperatingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, requirement } => {
                write!(
                    f,
                    "robust propeller operating {field} must be {requirement}"
                )
            }
            Self::ScenarioEvaluation {
                scenario_index,
                condition_index,
                error,
            } => write!(
                f,
                "robust propeller scenario {scenario_index} condition {condition_index:?} failed: {error}"
            ),
            Self::NonFiniteResult => write!(f, "robust propeller operating result is non-finite"),
        }
    }
}

impl std::error::Error for RobustOperatingError {}

/// Evaluate a deterministic source/condition uncertainty matrix.
pub fn evaluate_robust_operating_envelope(
    source: &PropellerSource,
    options: &RobustOperatingOptions,
) -> Result<RobustOperatingResult, RobustOperatingError> {
    validate_options(options)?;
    let mut scenario_results = Vec::with_capacity(options.scenarios.len());
    let mut all_admissible = true;
    let mut first_rejection = None;
    let mut worst_power: Option<(f64, usize, usize)> = None;
    let mut worst_efficiency: Option<(f64, usize, usize)> = None;

    for (scenario_index, scenario) in options.scenarios.iter().enumerate() {
        let mut scenario_source = source.clone();
        scenario_source.thrust *= scenario.thrust_scale;
        scenario_source.torque *= scenario.torque_scale;
        scenario_source.rpm *= scenario.rpm_scale;
        let envelope = match PropellerOperatingEnvelope::try_from_source(scenario_source) {
            Ok(envelope) => envelope,
            Err(error) => {
                all_admissible = false;
                first_rejection.get_or_insert((scenario_index, 0));
                scenario_results.push(RobustOperatingScenarioResult {
                    scenario: scenario.clone(),
                    operating_points: Vec::new(),
                    minimum_power_margin_w: None,
                    minimum_propulsive_efficiency: None,
                    admissible: false,
                    rejected_condition_index: None,
                    rejection: Some(error),
                });
                continue;
            }
        };

        let mut operating_points = Vec::with_capacity(options.conditions.len());
        let mut minimum_power_margin_w = None;
        let mut minimum_propulsive_efficiency = None;
        let mut rejection = None;
        let mut rejected_condition_index = None;
        for (condition_index, condition) in options.conditions.iter().enumerate() {
            let perturbed = PropellerOperatingCondition {
                freestream_m_s: condition.freestream_m_s * scenario.freestream_scale,
                rpm: condition.rpm * scenario.rpm_scale,
                inflow_angle_rad: condition.inflow_angle_rad + scenario.inflow_angle_offset_rad,
                density_kg_m3: condition.density_kg_m3 * scenario.density_scale,
                temperature_k: condition.temperature_k + scenario.temperature_offset_k,
            };
            match envelope.evaluate_condition(perturbed) {
                Ok(point) => {
                    minimum_power_margin_w = Some(
                        minimum_power_margin_w.map_or(point.power_margin_w, |value: f64| {
                            value.min(point.power_margin_w)
                        }),
                    );
                    minimum_propulsive_efficiency = Some(
                        minimum_propulsive_efficiency
                            .map_or(point.propulsive_efficiency, |value: f64| {
                                value.min(point.propulsive_efficiency)
                            }),
                    );
                    if let Some(value) = minimum_power_margin_w
                        && worst_power
                            .map(|(current, _, _)| value < current)
                            .unwrap_or(true)
                    {
                        worst_power = Some((value, scenario_index, condition_index));
                    }
                    if let Some(value) = minimum_propulsive_efficiency
                        && worst_efficiency
                            .map(|(current, _, _)| value < current)
                            .unwrap_or(true)
                    {
                        worst_efficiency = Some((value, scenario_index, condition_index));
                    }
                    operating_points.push(point);
                }
                Err(error @ PropellerOperatingEnvelopeError::ExceedsShaftPower { .. })
                | Err(error @ PropellerOperatingEnvelopeError::InsufficientHoverPower { .. }) => {
                    all_admissible = false;
                    first_rejection.get_or_insert((scenario_index, condition_index));
                    rejected_condition_index = Some(condition_index);
                    rejection = Some(error);
                    break;
                }
                Err(error) => {
                    return Err(RobustOperatingError::ScenarioEvaluation {
                        scenario_index,
                        condition_index: Some(condition_index),
                        error,
                    });
                }
            }
        }
        let admissible = rejection.is_none() && operating_points.len() == options.conditions.len();
        all_admissible &= admissible;
        scenario_results.push(RobustOperatingScenarioResult {
            scenario: scenario.clone(),
            operating_points,
            minimum_power_margin_w,
            minimum_propulsive_efficiency,
            admissible,
            rejected_condition_index,
            rejection,
        });
    }

    let (worst_case_scenario_index, worst_case_condition_index) = first_rejection
        .or_else(|| worst_power.map(|(_, scenario, condition)| (scenario, condition)))
        .or_else(|| worst_efficiency.map(|(_, scenario, condition)| (scenario, condition)))
        .unwrap_or((0, 0));
    let result = RobustOperatingResult {
        scenarios: scenario_results,
        all_admissible,
        worst_case_scenario_index,
        worst_case_condition_index,
        worst_power_margin_w: worst_power.map(|(value, _, _)| value),
        worst_propulsive_efficiency: worst_efficiency.map(|(value, _, _)| value),
    };
    if !result.is_finite() {
        return Err(RobustOperatingError::NonFiniteResult);
    }
    Ok(result)
}

fn scenario_is_finite(scenario: &PropellerUncertaintyScenario) -> bool {
    !scenario.label.is_empty()
        && [
            scenario.thrust_scale,
            scenario.torque_scale,
            scenario.rpm_scale,
            scenario.freestream_scale,
            scenario.density_scale,
            scenario.inflow_angle_offset_rad,
            scenario.temperature_offset_k,
        ]
        .iter()
        .all(|value| value.is_finite())
        && scenario.thrust_scale > 0.0
        && scenario.torque_scale > 0.0
        && scenario.rpm_scale > 0.0
        && scenario.freestream_scale > 0.0
        && scenario.density_scale > 0.0
}

fn operating_point_is_finite(point: &PropellerMultipointOperatingPoint) -> bool {
    [
        point.freestream_m_s,
        point.axial_freestream_m_s,
        point.inflow_angle_rad,
        point.rpm,
        point.density_kg_m3,
        point.temperature_k,
        point.speed_of_sound_m_s,
        point.advance_ratio,
        point.induced_velocity_m_s,
        point.disk_velocity_m_s,
        point.load_scale,
        point.thrust_n,
        point.useful_power_w,
        point.ideal_actuator_power_w,
        point.ideal_actuator_efficiency,
        point.propulsive_efficiency,
        point.power_margin_w,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn validate_options(options: &RobustOperatingOptions) -> Result<(), RobustOperatingError> {
    if options.scenarios.is_empty() {
        return Err(RobustOperatingError::InvalidConfiguration {
            field: "scenarios",
            requirement: "non-empty",
        });
    }
    if options.conditions.is_empty() {
        return Err(RobustOperatingError::InvalidConfiguration {
            field: "conditions",
            requirement: "non-empty",
        });
    }
    for scenario in &options.scenarios {
        if !scenario_is_finite(scenario) {
            return Err(RobustOperatingError::InvalidConfiguration {
                field: "scenario",
                requirement: "non-empty label and finite positive scales",
            });
        }
    }
    Ok(())
}
