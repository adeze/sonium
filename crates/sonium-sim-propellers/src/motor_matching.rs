//! Checked steady motor/propeller operating-point matching.
//!
//! The propeller load is evaluated with the crate's converged blade-element
//! momentum reference while a caller-supplied motor torque-speed curve is
//! linearly interpolated over the same RPM interval. A bracketed bisection
//! solve finds a positive-RPM torque balance without extrapolating either
//! model. This is a steady shaft-load contract; it does not model motor
//! electrical transients, controller dynamics, battery sag, or aeroelastic
//! deformation.

use std::fmt;

use crate::{
    BladeElementBemOptions, BladeElementBemResult, BladeElementError, BladeElementModel,
    BladeElementOperatingPoint,
};

/// One measured or caller-fitted motor torque-speed sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorTorqueSpeedPoint {
    /// Shaft speed [rev/min].
    pub rpm: f64,
    /// Motor torque available at this speed [N m].
    pub available_torque_nm: f64,
}

/// Checked piecewise-linear motor torque-speed curve.
#[derive(Debug, Clone, PartialEq)]
pub struct MotorTorqueSpeedCurve {
    points: Vec<MotorTorqueSpeedPoint>,
}

impl MotorTorqueSpeedCurve {
    /// Construct a curve from at least two finite, strictly ordered samples.
    pub fn try_new(points: Vec<MotorTorqueSpeedPoint>) -> Result<Self, MotorMatchingError> {
        if points.len() < 2 {
            return Err(MotorMatchingError::InvalidInput {
                field: "motor_points",
                requirement: "at least two points",
            });
        }

        for (index, point) in points.iter().enumerate() {
            if !point.rpm.is_finite() || point.rpm < 0.0 {
                return Err(MotorMatchingError::InvalidMotorPoint {
                    index,
                    field: "rpm",
                    requirement: "finite and non-negative",
                });
            }
            if !point.available_torque_nm.is_finite() || point.available_torque_nm < 0.0 {
                return Err(MotorMatchingError::InvalidMotorPoint {
                    index,
                    field: "available_torque_nm",
                    requirement: "finite and non-negative",
                });
            }
            if let Some(previous) = index.checked_sub(1).and_then(|i| points.get(i))
                && point.rpm <= previous.rpm
            {
                return Err(MotorMatchingError::NonIncreasingMotorRpm { index });
            }
        }

        Ok(Self { points })
    }

    /// Borrow curve samples in caller order.
    pub fn points(&self) -> &[MotorTorqueSpeedPoint] {
        &self.points
    }

    /// Evaluate the curve without extrapolating beyond its samples.
    pub fn torque_at_rpm(&self, rpm: f64) -> Result<f64, MotorMatchingError> {
        if !rpm.is_finite() || rpm < 0.0 {
            return Err(MotorMatchingError::InvalidInput {
                field: "rpm",
                requirement: "finite and non-negative",
            });
        }

        let minimum_rpm = self.points[0].rpm;
        let maximum_rpm = self.points[self.points.len() - 1].rpm;
        if rpm < minimum_rpm || rpm > maximum_rpm {
            return Err(MotorMatchingError::MotorCurveOutsideRange {
                rpm,
                minimum_rpm,
                maximum_rpm,
            });
        }

        if let Some(point) = self.points.iter().find(|point| point.rpm == rpm) {
            return Ok(point.available_torque_nm);
        }

        for window in self.points.windows(2) {
            let lower = window[0];
            let upper = window[1];
            if rpm < lower.rpm || rpm > upper.rpm {
                continue;
            }
            let fraction = (rpm - lower.rpm) / (upper.rpm - lower.rpm);
            let torque = lower.available_torque_nm
                + fraction * (upper.available_torque_nm - lower.available_torque_nm);
            if !torque.is_finite() {
                return Err(MotorMatchingError::NonFiniteResult {
                    field: "interpolated_motor_torque",
                });
            }
            return Ok(torque);
        }

        Err(MotorMatchingError::MotorCurveOutsideRange {
            rpm,
            minimum_rpm,
            maximum_rpm,
        })
    }
}

/// Uniform condition used while matching the motor to the propeller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorPropellerMatchCondition {
    /// Uniform axial freestream [m/s].
    pub freestream_m_s: f64,
    /// Air density [kg/m³].
    pub density_kg_m3: f64,
}

/// Controls the checked bracketed motor/propeller solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorPropellerMatchOptions {
    /// Lower positive RPM bracket [rev/min].
    pub rpm_min: f64,
    /// Upper RPM bracket [rev/min].
    pub rpm_max: f64,
    /// Maximum bisection updates.
    pub max_iterations: usize,
    /// Maximum BEM evaluations, including the two bracket endpoints.
    pub max_bem_evaluations: usize,
    /// Torque-balance convergence tolerance [N m].
    pub torque_tolerance_nm: f64,
    /// RPM bracket resolution [rev/min].
    pub rpm_tolerance: f64,
    /// BEM options used at each candidate RPM.
    pub bem_options: BladeElementBemOptions,
}

impl Default for MotorPropellerMatchOptions {
    fn default() -> Self {
        Self {
            rpm_min: 1_000.0,
            rpm_max: 20_000.0,
            max_iterations: 48,
            max_bem_evaluations: 64,
            torque_tolerance_nm: 1.0e-6,
            rpm_tolerance: 1.0e-3,
            bem_options: BladeElementBemOptions::default(),
        }
    }
}

/// Converged motor/propeller shaft operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct MotorPropellerMatchResult {
    /// BEM operating point at the torque balance.
    pub operating_point: BladeElementOperatingPoint,
    /// Uniform condition used for the match.
    pub condition: MotorPropellerMatchCondition,
    /// Motor torque interpolated at the reported RPM [N m].
    pub motor_torque_nm: f64,
    /// Propeller torque from the converged BEM result [N m].
    pub propeller_torque_nm: f64,
    /// Signed motor-minus-propeller torque residual [N m].
    pub torque_residual_nm: f64,
    /// Number of bisection updates accepted.
    pub iterations: usize,
    /// Number of BEM evaluations performed, including bracket endpoints.
    pub bem_evaluation_count: usize,
    /// Final enclosing RPM bracket [rev/min].
    pub bracket_rpm: [f64; 2],
    /// Converged BEM loads and sectional diagnostics.
    pub bem: BladeElementBemResult,
}

impl MotorPropellerMatchResult {
    /// Whether the result satisfies the finite, converged-state contract.
    pub fn is_finite(&self) -> bool {
        self.operating_point.rpm.is_finite()
            && self.operating_point.rpm > 0.0
            && self.condition.freestream_m_s.is_finite()
            && self.condition.freestream_m_s > 0.0
            && self.condition.density_kg_m3.is_finite()
            && self.condition.density_kg_m3 > 0.0
            && self.motor_torque_nm.is_finite()
            && self.motor_torque_nm >= 0.0
            && self.propeller_torque_nm.is_finite()
            && self.torque_residual_nm.is_finite()
            && self.bem_evaluation_count > 0
            && self.bracket_rpm[0].is_finite()
            && self.bracket_rpm[1].is_finite()
            && self.bracket_rpm[0] <= self.bracket_rpm[1]
            && self.bem.converged
            && self.bem.loads.operating_point == self.operating_point
            && self.bem.loads.torque_nm == self.propeller_torque_nm
    }
}

/// Failure-closed motor/propeller matching error.
#[derive(Debug, Clone, PartialEq)]
pub enum MotorMatchingError {
    /// A top-level scalar is outside its declared domain.
    InvalidInput {
        /// Input field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A motor curve point is invalid.
    InvalidMotorPoint {
        /// Point index.
        index: usize,
        /// Point field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Motor RPM samples are not strictly increasing.
    NonIncreasingMotorRpm {
        /// Index of the second invalid point.
        index: usize,
    },
    /// Requested RPM is outside the supplied motor curve.
    MotorCurveOutsideRange {
        /// Requested speed [rev/min].
        rpm: f64,
        /// Lowest curve speed [rev/min].
        minimum_rpm: f64,
        /// Highest curve speed [rev/min].
        maximum_rpm: f64,
    },
    /// The motor and propeller residuals have no sign-changing bracket.
    NoTorqueBracket {
        /// Lower endpoint residual [N m].
        lower_residual_nm: f64,
        /// Upper endpoint residual [N m].
        upper_residual_nm: f64,
    },
    /// The requested BEM work budget was exhausted.
    EvaluationBudgetExceeded {
        /// Number of evaluations that would be needed.
        requested: usize,
        /// Caller budget.
        budget: usize,
    },
    /// Bisection failed to meet both declared tolerances.
    NotConverged {
        /// Number of bisection updates performed.
        iterations: usize,
        /// Last torque residual [N m].
        residual_nm: f64,
        /// Last enclosing RPM bracket [rev/min].
        bracket_rpm: [f64; 2],
    },
    /// A nested BEM evaluation failed.
    BemEvaluationFailed {
        /// Candidate RPM [rev/min].
        rpm: f64,
        /// Underlying checked BEM error.
        error: Box<BladeElementError>,
    },
    /// An interpolation or final result became non-finite.
    NonFiniteResult {
        /// Result field name.
        field: &'static str,
    },
}

impl fmt::Display for MotorMatchingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, requirement } => {
                write!(f, "motor matching {field} must be {requirement}")
            }
            Self::InvalidMotorPoint {
                index,
                field,
                requirement,
            } => write!(f, "motor point {index} {field} must be {requirement}"),
            Self::NonIncreasingMotorRpm { index } => {
                write!(
                    f,
                    "motor RPM at index {index} must exceed the previous sample"
                )
            }
            Self::MotorCurveOutsideRange {
                rpm,
                minimum_rpm,
                maximum_rpm,
            } => write!(
                f,
                "motor curve does not cover {rpm:e} RPM; range is [{minimum_rpm:e}, {maximum_rpm:e}]"
            ),
            Self::NoTorqueBracket {
                lower_residual_nm,
                upper_residual_nm,
            } => write!(
                f,
                "motor/propeller torque residual has no bracket ({lower_residual_nm:e}, {upper_residual_nm:e})"
            ),
            Self::EvaluationBudgetExceeded { requested, budget } => write!(
                f,
                "motor matching needs {requested} BEM evaluations but budget is {budget}"
            ),
            Self::NotConverged {
                iterations,
                residual_nm,
                bracket_rpm,
            } => write!(
                f,
                "motor matching did not converge in {iterations} iterations (residual {residual_nm:e} N m, bracket [{}, {}] RPM)",
                bracket_rpm[0], bracket_rpm[1]
            ),
            Self::BemEvaluationFailed { rpm, error } => {
                write!(f, "BEM evaluation failed at {rpm:e} RPM: {error}")
            }
            Self::NonFiniteResult { field } => {
                write!(f, "motor matching result {field} is non-finite")
            }
        }
    }
}

impl std::error::Error for MotorMatchingError {}

#[derive(Debug, Clone)]
struct TorqueBalanceSample {
    rpm: f64,
    motor_torque_nm: f64,
    propeller_torque_nm: f64,
    residual_nm: f64,
    bem: BladeElementBemResult,
}

/// Find a steady RPM where motor and propeller torque balance.
pub fn evaluate_motor_propeller_match(
    model: &BladeElementModel,
    motor: &MotorTorqueSpeedCurve,
    condition: MotorPropellerMatchCondition,
    options: MotorPropellerMatchOptions,
) -> Result<MotorPropellerMatchResult, MotorMatchingError> {
    validate_condition(condition)?;
    validate_options(options)?;

    let mut bem_evaluation_count = 0;
    let lower = evaluate_balance(
        model,
        motor,
        condition,
        options,
        options.rpm_min,
        &mut bem_evaluation_count,
    )?;
    if lower.residual_nm.abs() <= options.torque_tolerance_nm {
        let rpm = lower.rpm;
        return build_result(lower, condition, bem_evaluation_count, 0, [rpm; 2]);
    }

    let upper = evaluate_balance(
        model,
        motor,
        condition,
        options,
        options.rpm_max,
        &mut bem_evaluation_count,
    )?;
    if upper.residual_nm.abs() <= options.torque_tolerance_nm {
        let rpm = upper.rpm;
        return build_result(upper, condition, bem_evaluation_count, 0, [rpm; 2]);
    }

    if (lower.residual_nm > 0.0) == (upper.residual_nm > 0.0) {
        return Err(MotorMatchingError::NoTorqueBracket {
            lower_residual_nm: lower.residual_nm,
            upper_residual_nm: upper.residual_nm,
        });
    }

    let mut low = lower;
    let mut high = upper;
    for iteration in 1..=options.max_iterations {
        let bracket_width = high.rpm - low.rpm;
        if !bracket_width.is_finite() || bracket_width <= options.rpm_tolerance {
            return Err(MotorMatchingError::NotConverged {
                iterations: iteration - 1,
                residual_nm: low.residual_nm.abs().min(high.residual_nm.abs()),
                bracket_rpm: [low.rpm, high.rpm],
            });
        }

        let midpoint_rpm = low.rpm + 0.5 * bracket_width;
        if midpoint_rpm <= low.rpm || midpoint_rpm >= high.rpm {
            return Err(MotorMatchingError::NotConverged {
                iterations: iteration - 1,
                residual_nm: low.residual_nm.abs().min(high.residual_nm.abs()),
                bracket_rpm: [low.rpm, high.rpm],
            });
        }

        let midpoint = evaluate_balance(
            model,
            motor,
            condition,
            options,
            midpoint_rpm,
            &mut bem_evaluation_count,
        )?;
        if midpoint.residual_nm.abs() <= options.torque_tolerance_nm {
            return build_result(
                midpoint,
                condition,
                bem_evaluation_count,
                iteration,
                [low.rpm, high.rpm],
            );
        }

        if (low.residual_nm > 0.0) == (midpoint.residual_nm > 0.0) {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }

    Err(MotorMatchingError::NotConverged {
        iterations: options.max_iterations,
        residual_nm: low.residual_nm.abs().min(high.residual_nm.abs()),
        bracket_rpm: [low.rpm, high.rpm],
    })
}

fn evaluate_balance(
    model: &BladeElementModel,
    motor: &MotorTorqueSpeedCurve,
    condition: MotorPropellerMatchCondition,
    options: MotorPropellerMatchOptions,
    rpm: f64,
    bem_evaluation_count: &mut usize,
) -> Result<TorqueBalanceSample, MotorMatchingError> {
    let motor_torque_nm = motor.torque_at_rpm(rpm)?;
    let requested = bem_evaluation_count.checked_add(1).ok_or(
        MotorMatchingError::EvaluationBudgetExceeded {
            requested: usize::MAX,
            budget: options.max_bem_evaluations,
        },
    )?;
    if requested > options.max_bem_evaluations {
        return Err(MotorMatchingError::EvaluationBudgetExceeded {
            requested,
            budget: options.max_bem_evaluations,
        });
    }

    let bem = model
        .evaluate_bem(
            BladeElementOperatingPoint {
                rpm,
                freestream_m_s: condition.freestream_m_s,
                density_kg_m3: condition.density_kg_m3,
            },
            options.bem_options,
        )
        .map_err(|error| MotorMatchingError::BemEvaluationFailed {
            rpm,
            error: Box::new(error),
        })?;
    *bem_evaluation_count = requested;
    let propeller_torque_nm = bem.loads.torque_nm;
    let residual_nm = motor_torque_nm - propeller_torque_nm;
    if !propeller_torque_nm.is_finite() {
        return Err(MotorMatchingError::NonFiniteResult {
            field: "propeller_torque_nm",
        });
    }
    if !residual_nm.is_finite() {
        return Err(MotorMatchingError::NonFiniteResult {
            field: "torque_residual_nm",
        });
    }

    Ok(TorqueBalanceSample {
        rpm,
        motor_torque_nm,
        propeller_torque_nm,
        residual_nm,
        bem,
    })
}

fn build_result(
    sample: TorqueBalanceSample,
    condition: MotorPropellerMatchCondition,
    bem_evaluation_count: usize,
    iterations: usize,
    bracket_rpm: [f64; 2],
) -> Result<MotorPropellerMatchResult, MotorMatchingError> {
    let result = MotorPropellerMatchResult {
        operating_point: sample.bem.loads.operating_point,
        condition,
        motor_torque_nm: sample.motor_torque_nm,
        propeller_torque_nm: sample.propeller_torque_nm,
        torque_residual_nm: sample.residual_nm,
        iterations,
        bem_evaluation_count,
        bracket_rpm,
        bem: sample.bem,
    };
    if !result.is_finite() {
        return Err(MotorMatchingError::NonFiniteResult {
            field: "motor_propeller_match",
        });
    }
    Ok(result)
}

fn validate_condition(condition: MotorPropellerMatchCondition) -> Result<(), MotorMatchingError> {
    if !condition.freestream_m_s.is_finite() || condition.freestream_m_s <= 0.0 {
        return Err(MotorMatchingError::InvalidInput {
            field: "freestream_m_s",
            requirement: "finite and positive",
        });
    }
    if !condition.density_kg_m3.is_finite() || condition.density_kg_m3 <= 0.0 {
        return Err(MotorMatchingError::InvalidInput {
            field: "density_kg_m3",
            requirement: "finite and positive",
        });
    }
    Ok(())
}

fn validate_options(options: MotorPropellerMatchOptions) -> Result<(), MotorMatchingError> {
    if !options.rpm_min.is_finite() || options.rpm_min <= 0.0 {
        return Err(MotorMatchingError::InvalidInput {
            field: "rpm_min",
            requirement: "finite and positive",
        });
    }
    if !options.rpm_max.is_finite() || options.rpm_max <= options.rpm_min {
        return Err(MotorMatchingError::InvalidInput {
            field: "rpm_max",
            requirement: "finite and greater than rpm_min",
        });
    }
    if options.max_iterations == 0 {
        return Err(MotorMatchingError::InvalidInput {
            field: "max_iterations",
            requirement: "non-zero",
        });
    }
    if options.max_bem_evaluations < 2 {
        return Err(MotorMatchingError::InvalidInput {
            field: "max_bem_evaluations",
            requirement: "at least two",
        });
    }
    for (field, value) in [
        ("torque_tolerance_nm", options.torque_tolerance_nm),
        ("rpm_tolerance", options.rpm_tolerance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(MotorMatchingError::InvalidInput {
                field,
                requirement: "finite and positive",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AirfoilPolar, AirfoilPolarPoint, BladeElementStation};
    use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

    fn model() -> BladeElementModel {
        let source = PropellerSource::drone_5inch(12_000.0);
        let stations = vec![
            BladeElementStation {
                radius_m: 0.018,
                chord_m: 0.018,
                pitch_deg: 24.0,
            },
            BladeElementStation {
                radius_m: 0.036,
                chord_m: 0.014,
                pitch_deg: 21.0,
            },
            BladeElementStation {
                radius_m: 0.054,
                chord_m: 0.010,
                pitch_deg: 18.0,
            },
        ];
        let polar = AirfoilPolar::try_new(vec![
            AirfoilPolarPoint {
                angle_of_attack_deg: -20.0,
                lift_coefficient: -0.8,
                drag_coefficient: 0.08,
            },
            AirfoilPolarPoint {
                angle_of_attack_deg: 0.0,
                lift_coefficient: 0.0,
                drag_coefficient: 0.02,
            },
            AirfoilPolarPoint {
                angle_of_attack_deg: 20.0,
                lift_coefficient: 0.8,
                drag_coefficient: 0.08,
            },
            AirfoilPolarPoint {
                angle_of_attack_deg: 40.0,
                lift_coefficient: 0.9,
                drag_coefficient: 0.16,
            },
        ])
        .expect("valid motor-matching polar");
        BladeElementModel::try_from_source(&source, stations, polar)
            .expect("valid motor-matching model")
    }

    fn condition() -> MotorPropellerMatchCondition {
        MotorPropellerMatchCondition {
            freestream_m_s: 10.0,
            density_kg_m3: 1.204,
        }
    }

    fn options() -> MotorPropellerMatchOptions {
        MotorPropellerMatchOptions {
            rpm_min: 6_000.0,
            rpm_max: 12_000.0,
            max_iterations: 48,
            max_bem_evaluations: 64,
            torque_tolerance_nm: 1.0e-7,
            rpm_tolerance: 1.0e-3,
            ..MotorPropellerMatchOptions::default()
        }
    }

    fn crossing_motor(model: &BladeElementModel) -> MotorTorqueSpeedCurve {
        let lower = model
            .evaluate_bem(
                BladeElementOperatingPoint {
                    rpm: options().rpm_min,
                    freestream_m_s: condition().freestream_m_s,
                    density_kg_m3: condition().density_kg_m3,
                },
                options().bem_options,
            )
            .expect("lower BEM endpoint");
        let upper = model
            .evaluate_bem(
                BladeElementOperatingPoint {
                    rpm: options().rpm_max,
                    freestream_m_s: condition().freestream_m_s,
                    density_kg_m3: condition().density_kg_m3,
                },
                options().bem_options,
            )
            .expect("upper BEM endpoint");
        assert!(upper.loads.torque_nm > 0.0);
        let scale = lower
            .loads
            .torque_nm
            .abs()
            .max(upper.loads.torque_nm.abs())
            .max(1.0e-3);
        MotorTorqueSpeedCurve::try_new(vec![
            MotorTorqueSpeedPoint {
                rpm: options().rpm_min,
                available_torque_nm: lower.loads.torque_nm + scale,
            },
            MotorTorqueSpeedPoint {
                rpm: options().rpm_max,
                available_torque_nm: 0.0,
            },
        ])
        .expect("valid crossing motor")
    }

    #[test]
    fn motor_curve_interpolates_without_extrapolation() {
        let curve = MotorTorqueSpeedCurve::try_new(vec![
            MotorTorqueSpeedPoint {
                rpm: 1_000.0,
                available_torque_nm: 0.4,
            },
            MotorTorqueSpeedPoint {
                rpm: 5_000.0,
                available_torque_nm: 0.2,
            },
            MotorTorqueSpeedPoint {
                rpm: 9_000.0,
                available_torque_nm: 0.0,
            },
        ])
        .expect("valid curve");
        assert_eq!(curve.torque_at_rpm(1_000.0).unwrap(), 0.4);
        assert!((curve.torque_at_rpm(7_000.0).unwrap() - 0.1).abs() < 1.0e-12);
        assert!(matches!(
            curve.torque_at_rpm(10_000.0),
            Err(MotorMatchingError::MotorCurveOutsideRange { .. })
        ));
    }

    #[test]
    fn motor_curve_rejects_malformed_points() {
        for points in [
            vec![MotorTorqueSpeedPoint {
                rpm: 1_000.0,
                available_torque_nm: 0.1,
            }],
            vec![
                MotorTorqueSpeedPoint {
                    rpm: 1_000.0,
                    available_torque_nm: 0.1,
                },
                MotorTorqueSpeedPoint {
                    rpm: 1_000.0,
                    available_torque_nm: 0.0,
                },
            ],
            vec![
                MotorTorqueSpeedPoint {
                    rpm: 1_000.0,
                    available_torque_nm: -0.1,
                },
                MotorTorqueSpeedPoint {
                    rpm: 2_000.0,
                    available_torque_nm: 0.0,
                },
            ],
        ] {
            assert!(MotorTorqueSpeedCurve::try_new(points).is_err());
        }
    }

    #[test]
    fn motor_match_is_deterministic_finite_and_torque_balanced() {
        let model = model();
        let motor = crossing_motor(&model);
        let first = evaluate_motor_propeller_match(&model, &motor, condition(), options())
            .expect("motor/propeller match");
        let second = evaluate_motor_propeller_match(&model, &motor, condition(), options())
            .expect("deterministic motor/propeller match");
        assert_eq!(first, second);
        assert!(first.is_finite());
        assert!(first.torque_residual_nm.abs() <= options().torque_tolerance_nm);
        assert!(first.operating_point.rpm >= options().rpm_min);
        assert!(first.operating_point.rpm <= options().rpm_max);
        assert!(first.bem_evaluation_count <= options().max_bem_evaluations);
    }

    #[test]
    fn motor_match_rejects_missing_bracket_and_curve_range() {
        let model = model();
        let no_bracket = MotorTorqueSpeedCurve::try_new(vec![
            MotorTorqueSpeedPoint {
                rpm: options().rpm_min,
                available_torque_nm: 1.0e6,
            },
            MotorTorqueSpeedPoint {
                rpm: options().rpm_max,
                available_torque_nm: 1.0e6,
            },
        ])
        .unwrap();
        assert!(matches!(
            evaluate_motor_propeller_match(&model, &no_bracket, condition(), options()),
            Err(MotorMatchingError::NoTorqueBracket { .. })
        ));

        let narrow = MotorTorqueSpeedCurve::try_new(vec![
            MotorTorqueSpeedPoint {
                rpm: 7_000.0,
                available_torque_nm: 0.1,
            },
            MotorTorqueSpeedPoint {
                rpm: 8_000.0,
                available_torque_nm: 0.0,
            },
        ])
        .unwrap();
        assert!(matches!(
            evaluate_motor_propeller_match(&model, &narrow, condition(), options()),
            Err(MotorMatchingError::MotorCurveOutsideRange { .. })
        ));
    }

    #[test]
    fn motor_match_rejects_invalid_condition_options_and_budget() {
        let model = model();
        let motor = crossing_motor(&model);
        assert!(matches!(
            evaluate_motor_propeller_match(
                &model,
                &motor,
                MotorPropellerMatchCondition {
                    freestream_m_s: 0.0,
                    ..condition()
                },
                options()
            ),
            Err(MotorMatchingError::InvalidInput {
                field: "freestream_m_s",
                ..
            })
        ));
        assert!(matches!(
            evaluate_motor_propeller_match(
                &model,
                &motor,
                condition(),
                MotorPropellerMatchOptions {
                    max_bem_evaluations: 1,
                    ..options()
                }
            ),
            Err(MotorMatchingError::InvalidInput {
                field: "max_bem_evaluations",
                ..
            })
        ));
        assert!(matches!(
            evaluate_motor_propeller_match(
                &model,
                &motor,
                condition(),
                MotorPropellerMatchOptions {
                    max_bem_evaluations: 2,
                    ..options()
                }
            ),
            Err(MotorMatchingError::EvaluationBudgetExceeded { .. })
        ));
    }

    #[test]
    fn motor_match_rejects_bem_non_convergence() {
        let model = model();
        let motor = crossing_motor(&model);
        let result = evaluate_motor_propeller_match(
            &model,
            &motor,
            condition(),
            MotorPropellerMatchOptions {
                bem_options: BladeElementBemOptions {
                    max_iterations: 1,
                    ..options().bem_options
                },
                ..options()
            },
        );
        assert!(matches!(
            result,
            Err(MotorMatchingError::BemEvaluationFailed { .. })
        ));
    }
}
