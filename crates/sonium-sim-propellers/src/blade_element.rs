//! Checked steady blade-element load reference.
//!
//! This module is a variable-load reference between the fixed-load operating
//! envelope and a resolved aerodynamic solver.  It integrates sectional lift
//! and drag over explicitly supplied radial stations at a prescribed axial
//! inflow.  It deliberately does not invent chord, pitch, or airfoil data and
//! does not claim induced-velocity, tip-loss, free-wake, or unsteady fidelity.
//! Those limitations are part of the public contract: callers must provide a
//! station geometry and a tabulated airfoil polar.  The direct evaluator is a
//! no-induction reference, while the BEM extension adds local steady axial and
//! tangential induction with Prandtl losses.  Neither result is a resolved-flow
//! claim: radial wake transport, unsteadiness, and installation effects remain
//! outside the model.

use std::f64::consts::PI;

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

/// One radial blade-element station.
///
/// `pitch_deg` is measured from the plane of rotation.  The station radius is
/// treated as a quadrature point; neighboring stations determine its radial
/// control-volume width, with the hub and tip radii as the outer boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementStation {
    /// Radius of the station [m].
    pub radius_m: f64,
    /// Local chord [m].
    pub chord_m: f64,
    /// Local blade pitch angle from the rotation plane [deg].
    pub pitch_deg: f64,
}

/// One point in a two-dimensional airfoil lift/drag polar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AirfoilPolarPoint {
    /// Angle of attack [deg].
    pub angle_of_attack_deg: f64,
    /// Section lift coefficient [-].
    pub lift_coefficient: f64,
    /// Section drag coefficient [-].
    pub drag_coefficient: f64,
}

/// A checked, linearly interpolated airfoil polar.
#[derive(Debug, Clone, PartialEq)]
pub struct AirfoilPolar {
    points: Vec<AirfoilPolarPoint>,
}

/// A steady axial-inflow operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementOperatingPoint {
    /// Rotation rate [rev/min], positive for the declared rotation sense.
    pub rpm: f64,
    /// Uniform axial freestream [m/s].
    pub freestream_m_s: f64,
    /// Air density [kg/m³].
    pub density_kg_m3: f64,
}

/// Integrated forces and local kinematics for one radial element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementStripLoad {
    /// Station radius [m].
    pub radius_m: f64,
    /// Radial control-volume width [m].
    pub width_m: f64,
    /// Local chord [m].
    pub chord_m: f64,
    /// Local pitch [deg].
    pub pitch_deg: f64,
    /// Relative-flow angle measured from the rotation plane [deg].
    pub inflow_angle_deg: f64,
    /// Section angle of attack [deg].
    pub angle_of_attack_deg: f64,
    /// Relative-flow speed [m/s].
    pub relative_speed_m_s: f64,
    /// Interpolated lift coefficient [-].
    pub lift_coefficient: f64,
    /// Interpolated drag coefficient [-].
    pub drag_coefficient: f64,
    /// Axial force per radial length from one blade [N/m].
    pub axial_force_per_m_n: f64,
    /// Tangential force per radial length from one blade [N/m].
    pub tangential_force_per_m_n: f64,
    /// Total thrust contribution from all blades [N].
    pub thrust_n: f64,
    /// Total torque contribution from all blades [N m].
    pub torque_nm: f64,
}

/// Integrated result from the steady blade-element reference.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeElementResult {
    /// Operating point used for the evaluation.
    pub operating_point: BladeElementOperatingPoint,
    /// Angular speed [rad/s].
    pub omega_rad_s: f64,
    /// Advance ratio `V / (n D)` [-].
    pub advance_ratio: f64,
    /// Tip speed [m/s].
    pub tip_speed_m_s: f64,
    /// Total axial thrust [N].
    pub thrust_n: f64,
    /// Total torque about the shaft [N m].
    pub torque_nm: f64,
    /// Signed shaft power `Q omega` [W].
    pub shaft_power_w: f64,
    /// Useful propulsive power `T V` [W].
    pub useful_power_w: f64,
    /// `T V / (Q omega)` when shaft power is positive, otherwise zero.
    pub propulsive_efficiency: f64,
    /// Per-station load and kinematic details in input order.
    pub strips: Vec<BladeElementStripLoad>,
}

/// Controls for the steady forward-flight blade-element/momentum iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementBemOptions {
    /// Maximum fixed-point iterations per radial station.
    pub max_iterations: usize,
    /// Maximum absolute induction update for convergence.
    pub convergence_tolerance: f64,
    /// Under-relaxation factor in `(0, 1]`.
    pub relaxation: f64,
    /// Apply Prandtl's finite-tip loss factor.
    pub include_tip_loss: bool,
    /// Apply Prandtl's finite-root loss factor.
    pub include_root_loss: bool,
}

impl Default for BladeElementBemOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_tolerance: 1.0e-8,
            relaxation: 0.35,
            include_tip_loss: true,
            include_root_loss: true,
        }
    }
}

/// External axial and tangential velocity corrections at one BEM station.
///
/// The corrections are added to the local velocities after the BEM induction
/// factors are applied. They are intended for bounded coupling to a separate
/// wake reference; they do not change the BEM momentum update itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementExternalInflow {
    /// Additional axial velocity along the rotor axis [m/s].
    pub axial_velocity_m_s: f64,
    /// Additional tangential velocity in the direction of positive rotation [m/s].
    pub tangential_velocity_m_s: f64,
}

impl BladeElementExternalInflow {
    const ZERO: Self = Self {
        axial_velocity_m_s: 0.0,
        tangential_velocity_m_s: 0.0,
    };
}

/// Induction diagnostics for one BEM radial station.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementBemStrip {
    /// Axial induction `a`, with axial velocity `V (1 + a)`.
    pub axial_induction: f64,
    /// Tangential induction `a'`, with tangential velocity `Ωr (1 - a')`.
    pub tangential_induction: f64,
    /// Combined Prandtl loss factor used by the update [-].
    pub loss_factor: f64,
    /// Number of fixed-point iterations used.
    pub iterations: usize,
    /// Final absolute induction update residual.
    pub residual: f64,
}

/// Converged forward-flight blade-element/momentum result.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeElementBemResult {
    /// The integrated loads evaluated using the converged induced velocities.
    pub loads: BladeElementResult,
    /// Per-station induction and loss diagnostics in station order.
    pub strips: Vec<BladeElementBemStrip>,
    /// Whether every station met the requested tolerance.
    pub converged: bool,
    /// Largest iteration count among stations.
    pub max_iterations_used: usize,
}

/// One level in a BEM radial-load convergence ladder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementBemConvergenceLevel {
    /// Number of caller-supplied radial stations at this level.
    pub station_count: usize,
    /// Integrated thrust [N].
    pub thrust_n: f64,
    /// Integrated torque [N m].
    pub torque_nm: f64,
    /// Integrated shaft power [W].
    pub shaft_power_w: f64,
    /// Relative thrust change from the preceding level, if one exists.
    pub relative_thrust_change: Option<f64>,
    /// Relative torque change from the preceding level, if one exists.
    pub relative_torque_change: Option<f64>,
    /// Whether both load changes meet the requested convergence tolerance.
    pub converged: bool,
}

/// BEM results and load-change evidence for an explicit radial refinement ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeElementBemConvergence {
    /// Per-level load metrics in the exact caller-supplied order.
    pub levels: Vec<BladeElementBemConvergenceLevel>,
    /// Full BEM result for each level, in the same order as [`Self::levels`].
    pub results: Vec<BladeElementBemResult>,
    /// Whether the final level met the requested relative load tolerance.
    pub converged: bool,
}

/// Options for comparing the direct no-induction and steady BEM references.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementFidelityOptions {
    /// Absolute floor used in relative-difference denominators.
    pub relative_scale_floor: f64,
    /// Relative tolerance below which two candidate metric values are an
    /// ambiguous tie for strict ranking evidence.
    pub rank_tie_tolerance: f64,
}

impl Default for BladeElementFidelityOptions {
    fn default() -> Self {
        Self {
            relative_scale_floor: 1.0e-12,
            rank_tie_tolerance: 1.0e-12,
        }
    }
}

/// Scalar used when checking whether direct and BEM candidate order agrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BladeElementRankMetric {
    /// Total axial thrust [N].
    Thrust,
    /// Total shaft torque [N m].
    Torque,
    /// Mechanical shaft input [W].
    ShaftPower,
    /// Propulsive efficiency [-].
    PropulsiveEfficiency,
}

impl std::fmt::Display for BladeElementRankMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Thrust => "thrust",
            Self::Torque => "torque",
            Self::ShaftPower => "shaft-power",
            Self::PropulsiveEfficiency => "propulsive-efficiency",
        };
        f.write_str(name)
    }
}

/// Direction used for direct-vs-BEM candidate ranking evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BladeElementRankOrder {
    /// Smaller metric values are preferred.
    Ascending,
    /// Larger metric values are preferred.
    Descending,
}

/// Relative differences between the two checked blade-element references at
/// one explicitly associated operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeElementFidelityComparison {
    /// Operating point shared by both results.
    pub operating_point: BladeElementOperatingPoint,
    /// Direct no-induction thrust [N].
    pub direct_thrust_n: f64,
    /// BEM thrust [N].
    pub bem_thrust_n: f64,
    /// Relative thrust difference.
    pub relative_thrust_difference: f64,
    /// Direct no-induction torque [N m].
    pub direct_torque_nm: f64,
    /// BEM torque [N m].
    pub bem_torque_nm: f64,
    /// Relative torque difference.
    pub relative_torque_difference: f64,
    /// Direct no-induction shaft power [W].
    pub direct_shaft_power_w: f64,
    /// BEM shaft power [W].
    pub bem_shaft_power_w: f64,
    /// Relative shaft-power difference.
    pub relative_shaft_power_difference: f64,
    /// Direct no-induction propulsive efficiency [-].
    pub direct_propulsive_efficiency: f64,
    /// BEM propulsive efficiency [-].
    pub bem_propulsive_efficiency: f64,
    /// Relative propulsive-efficiency difference.
    pub relative_propulsive_efficiency_difference: f64,
}

/// Ordered-candidate evidence for whether direct and BEM rankings agree.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeElementRankEvidence {
    /// Operating point shared by every candidate in both sets.
    pub operating_point: BladeElementOperatingPoint,
    /// Metric used for the ranking.
    pub metric: BladeElementRankMetric,
    /// Preferred order used to construct both index lists.
    pub order: BladeElementRankOrder,
    /// Candidate indices from best to worst under the direct reference.
    pub direct_candidate_order: Vec<usize>,
    /// Candidate indices from best to worst under the BEM reference.
    pub bem_candidate_order: Vec<usize>,
    /// Whether the two strict orderings are identical.
    pub preserved: bool,
}

/// Failure-closed direct-vs-BEM comparison or ranking error.
#[derive(Debug, Clone, PartialEq)]
pub enum BladeElementFidelityError {
    /// A comparison option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A result set was empty where candidate evidence was requested.
    EmptyResults {
        /// Result-set field name.
        field: &'static str,
    },
    /// The direct and BEM result sets do not describe the same candidates.
    ResultCountMismatch {
        /// Number of direct results.
        direct_count: usize,
        /// Number of BEM results.
        bem_count: usize,
    },
    /// Strict ranking needs at least two candidates.
    InsufficientCandidates {
        /// Number of supplied candidates.
        count: usize,
    },
    /// A paired direct/BEM result has different point metadata.
    OperatingPointMismatch {
        /// Candidate or multipoint index.
        index: usize,
    },
    /// A candidate set contains multiple operating points, so its rank is not
    /// an apples-to-apples comparison.
    CandidateOperatingPointMismatch {
        /// Candidate index whose point differs from the first candidate.
        index: usize,
    },
    /// A result carries an invalid operating-point metadata record.
    InvalidOperatingPoint {
        /// `direct` or `bem` result family.
        model: &'static str,
        /// Candidate or multipoint index.
        index: usize,
    },
    /// A BEM result was not marked converged.
    UnconvergedBem {
        /// Candidate or multipoint index.
        index: usize,
    },
    /// A result metric or operating-point scalar is non-finite or invalid.
    NonFiniteMetric {
        /// `direct` or `bem` result family.
        model: &'static str,
        /// Candidate or multipoint index.
        index: usize,
        /// Metric that failed validation.
        metric: BladeElementRankMetric,
    },
    /// Relative subtraction overflowed despite finite source metrics.
    NonFiniteDifference {
        /// Comparison field name.
        field: &'static str,
    },
    /// A strict rank is ambiguous within the configured tie tolerance.
    AmbiguousRank {
        /// `direct` or `bem` result family.
        model: &'static str,
        /// First candidate in the ambiguous pair.
        first_candidate: usize,
        /// Second candidate in the ambiguous pair.
        second_candidate: usize,
        /// Metric that tied.
        metric: BladeElementRankMetric,
        /// Relative difference between the two values.
        relative_difference: f64,
        /// Configured tie tolerance.
        tolerance: f64,
    },
}

impl std::fmt::Display for BladeElementFidelityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    f,
                    "blade-element fidelity option {field} must be {requirement}"
                )
            }
            Self::EmptyResults { field } => {
                write!(f, "blade-element fidelity {field} must be non-empty")
            }
            Self::ResultCountMismatch {
                direct_count,
                bem_count,
            } => write!(
                f,
                "blade-element fidelity result counts differ: direct={direct_count}, bem={bem_count}"
            ),
            Self::InsufficientCandidates { count } => write!(
                f,
                "blade-element fidelity ranking requires at least two candidates, got {count}"
            ),
            Self::OperatingPointMismatch { index } => write!(
                f,
                "blade-element fidelity operating point mismatch at index {index}"
            ),
            Self::CandidateOperatingPointMismatch { index } => write!(
                f,
                "blade-element fidelity candidate operating point mismatch at index {index}"
            ),
            Self::InvalidOperatingPoint { model, index } => write!(
                f,
                "blade-element fidelity {model} result {index} has invalid operating-point metadata"
            ),
            Self::UnconvergedBem { index } => {
                write!(
                    f,
                    "blade-element fidelity BEM result {index} is not converged"
                )
            }
            Self::NonFiniteMetric {
                model,
                index,
                metric,
            } => write!(
                f,
                "blade-element fidelity {model} result {index} has invalid {metric}"
            ),
            Self::NonFiniteDifference { field } => write!(
                f,
                "blade-element fidelity relative {field} difference is non-finite"
            ),
            Self::AmbiguousRank {
                model,
                first_candidate,
                second_candidate,
                metric,
                relative_difference,
                tolerance,
            } => write!(
                f,
                "blade-element fidelity {model} {metric} rank is ambiguous for candidates {first_candidate}/{second_candidate}: relative difference {relative_difference:e} <= {tolerance:e}"
            ),
        }
    }
}

impl std::error::Error for BladeElementFidelityError {}

/// Failure-closed input or evaluation error for the blade-element reference.
#[derive(Debug, Clone, PartialEq)]
pub enum BladeElementError {
    /// A source or operating-point scalar is outside its physical domain.
    InvalidInput {
        /// Input field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A station scalar is invalid.
    InvalidStation {
        /// Station index in the caller-provided array.
        index: usize,
        /// Input field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Station radii are not strictly increasing.
    NonIncreasingStationRadius {
        /// Index of the second station in the invalid pair.
        index: usize,
    },
    /// A polar point scalar is invalid.
    InvalidPolarPoint {
        /// Polar point index in the caller-provided array.
        index: usize,
        /// Input field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Polar angle samples are not strictly increasing.
    NonIncreasingPolarAngle {
        /// Index of the second polar point in the invalid pair.
        index: usize,
    },
    /// A direct polar query is outside the tabulated range.
    PolarQueryOutsideRange {
        /// Requested angle [deg].
        angle_of_attack_deg: f64,
        /// Lower tabulated polar bound [deg].
        minimum_deg: f64,
        /// Upper tabulated polar bound [deg].
        maximum_deg: f64,
    },
    /// The requested section angle is outside the supplied polar.
    AngleOfAttackOutsidePolar {
        /// Station index that requested the out-of-range angle.
        station: usize,
        /// Requested angle [deg].
        angle_of_attack_deg: f64,
        /// Lower tabulated polar bound [deg].
        minimum_deg: f64,
        /// Upper tabulated polar bound [deg].
        maximum_deg: f64,
    },
    /// An evaluated result became non-finite.
    NonFiniteResult {
        /// Result field name.
        field: &'static str,
    },
    /// BEM options are invalid.
    InvalidBemOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The external inflow array does not correspond to the model stations.
    ExternalInflowCountMismatch {
        /// Number of model stations.
        model_count: usize,
        /// Number of supplied external inflow stations.
        inflow_count: usize,
    },
    /// An external inflow component is invalid.
    InvalidExternalInflow {
        /// Station index in the external inflow array.
        station: usize,
        /// Invalid component name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The forward-flight momentum equations are singular at zero axial speed.
    BemRequiresPositiveFreestream,
    /// A BEM station update left the declared induction domain.
    InductionOutOfRange {
        /// Station index.
        station: usize,
        /// Induction field name.
        field: &'static str,
    },
    /// A BEM station update encountered a singular force equation.
    InductionSingularity {
        /// Station index.
        station: usize,
        /// Equation name.
        equation: &'static str,
    },
    /// A BEM station did not converge within the configured iteration budget.
    InductionNotConverged {
        /// Station index.
        station: usize,
        /// Number of iterations performed.
        iterations: usize,
        /// Final absolute update residual.
        residual: f64,
    },
    /// One point in a multipoint BEM evaluation failed.
    MultipointEvaluationFailed {
        /// Input point index, preserving caller order.
        point_index: usize,
        /// Failure from that point's checked evaluation.
        error: Box<BladeElementError>,
    },
    /// One radial-refinement level failed.
    ConvergenceLevelFailed {
        /// Refinement level index, preserving caller order.
        level_index: usize,
        /// Failure from that level's checked BEM evaluation.
        error: Box<BladeElementError>,
    },
}

impl std::fmt::Display for BladeElementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { field, requirement } => {
                write!(f, "blade-element {field} must be {requirement}")
            }
            Self::InvalidStation {
                index,
                field,
                requirement,
            } => write!(
                f,
                "blade-element station {index} {field} must be {requirement}"
            ),
            Self::NonIncreasingStationRadius { index } => write!(
                f,
                "blade-element station radius at index {index} must exceed the previous radius"
            ),
            Self::InvalidPolarPoint {
                index,
                field,
                requirement,
            } => write!(
                f,
                "blade-element polar point {index} {field} must be {requirement}"
            ),
            Self::NonIncreasingPolarAngle { index } => write!(
                f,
                "blade-element polar angle at index {index} must exceed the previous angle"
            ),
            Self::PolarQueryOutsideRange {
                angle_of_attack_deg,
                minimum_deg,
                maximum_deg,
            } => write!(
                f,
                "blade-element polar query {angle_of_attack_deg:e} deg is outside [{minimum_deg:e}, {maximum_deg:e}] deg"
            ),
            Self::AngleOfAttackOutsidePolar {
                station,
                angle_of_attack_deg,
                minimum_deg,
                maximum_deg,
            } => write!(
                f,
                "blade-element station {station} angle of attack {angle_of_attack_deg:e} deg is outside polar [{minimum_deg:e}, {maximum_deg:e}] deg"
            ),
            Self::NonFiniteResult { field } => {
                write!(f, "blade-element result {field} is non-finite")
            }
            Self::InvalidBemOptions { field, requirement } => {
                write!(f, "blade-element BEM option {field} must be {requirement}")
            }
            Self::ExternalInflowCountMismatch {
                model_count,
                inflow_count,
            } => write!(
                f,
                "blade-element external inflow station counts differ: model={model_count}, inflow={inflow_count}"
            ),
            Self::InvalidExternalInflow {
                station,
                field,
                requirement,
            } => write!(
                f,
                "blade-element external inflow station {station} {field} must be {requirement}"
            ),
            Self::BemRequiresPositiveFreestream => write!(
                f,
                "blade-element BEM requires a strictly positive axial freestream"
            ),
            Self::InductionOutOfRange { station, field } => write!(
                f,
                "blade-element BEM station {station} {field} left the supported induction range"
            ),
            Self::InductionSingularity { station, equation } => write!(
                f,
                "blade-element BEM station {station} has a singular {equation} equation"
            ),
            Self::InductionNotConverged {
                station,
                iterations,
                residual,
            } => write!(
                f,
                "blade-element BEM station {station} did not converge in {iterations} iterations (residual {residual:e})"
            ),
            Self::MultipointEvaluationFailed { point_index, error } => write!(
                f,
                "blade-element BEM multipoint {point_index} failed: {error}"
            ),
            Self::ConvergenceLevelFailed { level_index, error } => write!(
                f,
                "blade-element BEM convergence level {level_index} failed: {error}"
            ),
        }
    }
}

impl std::error::Error for BladeElementError {}

impl AirfoilPolar {
    /// Construct a checked polar from strictly increasing angle samples.
    pub fn try_new(points: Vec<AirfoilPolarPoint>) -> Result<Self, BladeElementError> {
        if points.len() < 2 {
            return Err(BladeElementError::InvalidInput {
                field: "polar_points",
                requirement: "at least two points",
            });
        }
        for (index, point) in points.iter().enumerate() {
            if !point.angle_of_attack_deg.is_finite() {
                return Err(BladeElementError::InvalidPolarPoint {
                    index,
                    field: "angle_of_attack_deg",
                    requirement: "finite",
                });
            }
            if !point.lift_coefficient.is_finite() {
                return Err(BladeElementError::InvalidPolarPoint {
                    index,
                    field: "lift_coefficient",
                    requirement: "finite",
                });
            }
            if !point.drag_coefficient.is_finite() || point.drag_coefficient < 0.0 {
                return Err(BladeElementError::InvalidPolarPoint {
                    index,
                    field: "drag_coefficient",
                    requirement: "finite and non-negative",
                });
            }
            if let Some(previous) = index.checked_sub(1).and_then(|i| points.get(i))
                && point.angle_of_attack_deg <= previous.angle_of_attack_deg
            {
                return Err(BladeElementError::NonIncreasingPolarAngle { index });
            }
        }
        Ok(Self { points })
    }

    /// Borrow the checked polar points in input order.
    pub fn points(&self) -> &[AirfoilPolarPoint] {
        &self.points
    }

    /// Interpolate lift and drag at an angle of attack.
    pub fn coefficients(&self, angle_of_attack_deg: f64) -> Result<(f64, f64), BladeElementError> {
        if !angle_of_attack_deg.is_finite() {
            return Err(BladeElementError::InvalidInput {
                field: "angle_of_attack_deg",
                requirement: "finite",
            });
        }
        let minimum_deg = self.points[0].angle_of_attack_deg;
        let maximum_deg = self.points[self.points.len() - 1].angle_of_attack_deg;
        if !(minimum_deg..=maximum_deg).contains(&angle_of_attack_deg) {
            return Err(BladeElementError::PolarQueryOutsideRange {
                angle_of_attack_deg,
                minimum_deg,
                maximum_deg,
            });
        }
        if angle_of_attack_deg == maximum_deg {
            let point = self.points[self.points.len() - 1];
            return Ok((point.lift_coefficient, point.drag_coefficient));
        }
        for pair in self.points.windows(2) {
            let [lower, upper] = pair else { unreachable!() };
            if angle_of_attack_deg <= upper.angle_of_attack_deg {
                let fraction = (angle_of_attack_deg - lower.angle_of_attack_deg)
                    / (upper.angle_of_attack_deg - lower.angle_of_attack_deg);
                return Ok((
                    lower.lift_coefficient
                        + fraction * (upper.lift_coefficient - lower.lift_coefficient),
                    lower.drag_coefficient
                        + fraction * (upper.drag_coefficient - lower.drag_coefficient),
                ));
            }
        }
        unreachable!("validated polar must bracket a finite in-range angle")
    }
}

/// Checked steady blade-element model with explicit geometry and polar data.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeElementModel {
    n_blades: usize,
    hub_radius_m: f64,
    tip_radius_m: f64,
    default_rpm: f64,
    default_density_kg_m3: f64,
    stations: Vec<BladeElementStation>,
    polar: AirfoilPolar,
}

impl BladeElementModel {
    /// Construct a model from a source's blade count/radii and explicit input data.
    pub fn try_from_source(
        source: &PropellerSource,
        stations: Vec<BladeElementStation>,
        polar: AirfoilPolar,
    ) -> Result<Self, BladeElementError> {
        for (field, value) in [
            ("source.rpm", source.rpm),
            ("source.density", source.density),
            ("source.hub_radius", source.hub_radius),
            ("source.tip_radius", source.tip_radius),
        ] {
            if !value.is_finite() {
                return Err(BladeElementError::InvalidInput {
                    field,
                    requirement: "finite",
                });
            }
        }
        if source.n_blades == 0 {
            return Err(BladeElementError::InvalidInput {
                field: "source.n_blades",
                requirement: "non-zero",
            });
        }
        if source.rpm <= 0.0 {
            return Err(BladeElementError::InvalidInput {
                field: "source.rpm",
                requirement: "positive",
            });
        }
        if source.density <= 0.0 {
            return Err(BladeElementError::InvalidInput {
                field: "source.density",
                requirement: "positive",
            });
        }
        if source.hub_radius < 0.0 || source.tip_radius <= source.hub_radius {
            return Err(BladeElementError::InvalidInput {
                field: "source.hub_radius/source.tip_radius",
                requirement: "0 <= hub_radius < tip_radius",
            });
        }
        if stations.is_empty() {
            return Err(BladeElementError::InvalidInput {
                field: "stations",
                requirement: "non-empty",
            });
        }
        for (index, station) in stations.iter().enumerate() {
            if !station.radius_m.is_finite() {
                return Err(BladeElementError::InvalidStation {
                    index,
                    field: "radius_m",
                    requirement: "finite",
                });
            }
            if station.radius_m <= source.hub_radius || station.radius_m >= source.tip_radius {
                return Err(BladeElementError::InvalidStation {
                    index,
                    field: "radius_m",
                    requirement: "strictly inside hub/tip radii",
                });
            }
            if !station.chord_m.is_finite() || station.chord_m <= 0.0 {
                return Err(BladeElementError::InvalidStation {
                    index,
                    field: "chord_m",
                    requirement: "finite and positive",
                });
            }
            if !station.pitch_deg.is_finite() {
                return Err(BladeElementError::InvalidStation {
                    index,
                    field: "pitch_deg",
                    requirement: "finite",
                });
            }
            if let Some(previous) = index.checked_sub(1).and_then(|i| stations.get(i))
                && station.radius_m <= previous.radius_m
            {
                return Err(BladeElementError::NonIncreasingStationRadius { index });
            }
        }
        Ok(Self {
            n_blades: source.n_blades,
            hub_radius_m: source.hub_radius,
            tip_radius_m: source.tip_radius,
            default_rpm: source.rpm,
            default_density_kg_m3: source.density,
            stations,
            polar,
        })
    }

    /// Borrow the explicit stations used by this model.
    pub fn stations(&self) -> &[BladeElementStation] {
        &self.stations
    }

    /// Number of blades represented by this checked model.
    pub fn n_blades(&self) -> usize {
        self.n_blades
    }

    /// Hub radius used by this checked model [m].
    pub fn hub_radius_m(&self) -> f64 {
        self.hub_radius_m
    }

    /// Tip radius used by this checked model [m].
    pub fn tip_radius_m(&self) -> f64 {
        self.tip_radius_m
    }

    /// Borrow the explicit airfoil polar used by this model.
    pub fn polar(&self) -> &AirfoilPolar {
        &self.polar
    }

    /// Evaluate using the source RPM/density and a supplied axial freestream.
    pub fn evaluate_at_freestream(
        &self,
        freestream_m_s: f64,
    ) -> Result<BladeElementResult, BladeElementError> {
        self.evaluate(BladeElementOperatingPoint {
            rpm: self.default_rpm,
            freestream_m_s,
            density_kg_m3: self.default_density_kg_m3,
        })
    }

    /// Evaluate at an explicitly checked RPM, axial freestream, and density.
    ///
    /// Strip thrust/torque accumulate in deterministic root-to-tip station
    /// order with naive `+=` summation (worst-case error ~`n·eps·max strip`,
    /// negligible for the station counts used here); each strip is checked
    /// finite before it is added, so a corrupt strip errors instead of
    /// poisoning the totals.
    pub fn evaluate(
        &self,
        operating_point: BladeElementOperatingPoint,
    ) -> Result<BladeElementResult, BladeElementError> {
        validate_operating_point(operating_point)?;
        let omega_rad_s = 2.0 * PI * operating_point.rpm / 60.0;
        let tip_speed_m_s = omega_rad_s * self.tip_radius_m;
        let diameter_m = 2.0 * self.tip_radius_m;
        let revolutions_per_second = operating_point.rpm / 60.0;
        let advance_ratio = operating_point.freestream_m_s / (revolutions_per_second * diameter_m);
        let widths = self.station_widths();
        let blade_count = self.n_blades as f64;
        let mut strips = Vec::with_capacity(self.stations.len());
        let mut thrust_n = 0.0;
        let mut torque_nm = 0.0;

        for (index, (station, width_m)) in self.stations.iter().zip(widths).enumerate() {
            let tangential_speed_m_s = omega_rad_s * station.radius_m;
            let section = evaluate_section(
                &self.polar,
                station,
                operating_point,
                operating_point.freestream_m_s,
                tangential_speed_m_s,
                index,
            )?;
            let strip_thrust_n = blade_count * section.axial_force_per_m_n * width_m;
            let strip_torque_nm =
                blade_count * section.tangential_force_per_m_n * station.radius_m * width_m;
            if !strip_thrust_n.is_finite() || !strip_torque_nm.is_finite() {
                return Err(BladeElementError::NonFiniteResult {
                    field: "strip_load",
                });
            }
            thrust_n += strip_thrust_n;
            torque_nm += strip_torque_nm;
            strips.push(BladeElementStripLoad {
                radius_m: station.radius_m,
                width_m,
                chord_m: station.chord_m,
                pitch_deg: station.pitch_deg,
                inflow_angle_deg: section.inflow_angle_deg,
                angle_of_attack_deg: section.angle_of_attack_deg,
                relative_speed_m_s: section.relative_speed_m_s,
                lift_coefficient: section.lift_coefficient,
                drag_coefficient: section.drag_coefficient,
                axial_force_per_m_n: section.axial_force_per_m_n,
                tangential_force_per_m_n: section.tangential_force_per_m_n,
                thrust_n: strip_thrust_n,
                torque_nm: strip_torque_nm,
            });
        }

        let shaft_power_w = torque_nm * omega_rad_s;
        let useful_power_w = thrust_n * operating_point.freestream_m_s;
        let propulsive_efficiency = if shaft_power_w > 0.0 {
            useful_power_w / shaft_power_w
        } else {
            0.0
        };
        for (field, value) in [
            ("thrust_n", thrust_n),
            ("torque_nm", torque_nm),
            ("shaft_power_w", shaft_power_w),
            ("useful_power_w", useful_power_w),
            ("propulsive_efficiency", propulsive_efficiency),
        ] {
            if !value.is_finite() {
                return Err(BladeElementError::NonFiniteResult { field });
            }
        }
        Ok(BladeElementResult {
            operating_point,
            omega_rad_s,
            advance_ratio,
            tip_speed_m_s,
            thrust_n,
            torque_nm,
            shaft_power_w,
            useful_power_w,
            propulsive_efficiency,
            strips,
        })
    }

    /// Evaluate a steady forward-flight blade-element/momentum reference.
    ///
    /// The solve iterates axial and tangential induction independently at
    /// each station and applies Prandtl finite-tip/root loss factors.  It is a
    /// local, steady BEM reference: no radial wake transport, skew correction,
    /// dynamic stall, or unsteady wake is represented.  A strictly positive
    /// freestream is required because the axial momentum update is singular
    /// in hover; use [`Self::evaluate_at_freestream`] for the non-induced
    /// hover reference.
    pub fn evaluate_bem(
        &self,
        operating_point: BladeElementOperatingPoint,
        options: BladeElementBemOptions,
    ) -> Result<BladeElementBemResult, BladeElementError> {
        let external_inflow = vec![BladeElementExternalInflow::ZERO; self.stations.len()];
        self.evaluate_bem_with_external_inflow(operating_point, options, &external_inflow)
    }

    /// Evaluate BEM with caller-supplied steady external station velocities.
    ///
    /// The external corrections are added to the induced BEM local velocities
    /// during both the induction iteration and final load evaluation. This is
    /// a bounded coupling contract for reference wake models; it does not
    /// claim unsteady shedding, dynamic stall, or aeroelastic fidelity.
    pub fn evaluate_bem_with_external_inflow(
        &self,
        operating_point: BladeElementOperatingPoint,
        options: BladeElementBemOptions,
        external_inflow: &[BladeElementExternalInflow],
    ) -> Result<BladeElementBemResult, BladeElementError> {
        validate_operating_point(operating_point)?;
        validate_bem_options(options)?;
        if external_inflow.len() != self.stations.len() {
            return Err(BladeElementError::ExternalInflowCountMismatch {
                model_count: self.stations.len(),
                inflow_count: external_inflow.len(),
            });
        }
        for (station, inflow) in external_inflow.iter().enumerate() {
            if !inflow.axial_velocity_m_s.is_finite() {
                return Err(BladeElementError::InvalidExternalInflow {
                    station,
                    field: "axial_velocity_m_s",
                    requirement: "finite",
                });
            }
            if !inflow.tangential_velocity_m_s.is_finite() {
                return Err(BladeElementError::InvalidExternalInflow {
                    station,
                    field: "tangential_velocity_m_s",
                    requirement: "finite",
                });
            }
        }
        if operating_point.freestream_m_s <= 0.0 {
            return Err(BladeElementError::BemRequiresPositiveFreestream);
        }

        let omega_rad_s = 2.0 * PI * operating_point.rpm / 60.0;
        let tip_speed_m_s = omega_rad_s * self.tip_radius_m;
        let diameter_m = 2.0 * self.tip_radius_m;
        let revolutions_per_second = operating_point.rpm / 60.0;
        let advance_ratio = operating_point.freestream_m_s / (revolutions_per_second * diameter_m);
        let widths = self.station_widths();
        let blade_count = self.n_blades as f64;
        let mut loads = Vec::with_capacity(self.stations.len());
        let mut induction = Vec::with_capacity(self.stations.len());
        let mut thrust_n = 0.0;
        let mut torque_nm = 0.0;
        let mut max_iterations_used = 0;

        for (index, (station, width_m)) in self.stations.iter().zip(widths).enumerate() {
            let external = external_inflow[index];
            let solidity = blade_count * station.chord_m / (2.0 * PI * station.radius_m);
            let mut axial_induction = 0.0;
            let mut tangential_induction = 0.0;
            let mut residual = f64::INFINITY;
            let mut iterations_used = 0;
            let mut loss_factor = 1.0;

            for iteration in 1..=options.max_iterations {
                let axial_velocity_m_s = operating_point.freestream_m_s * (1.0 + axial_induction)
                    + external.axial_velocity_m_s;
                let tangential_velocity_m_s =
                    omega_rad_s * station.radius_m * (1.0 - tangential_induction)
                        + external.tangential_velocity_m_s;
                let section = evaluate_section(
                    &self.polar,
                    station,
                    operating_point,
                    axial_velocity_m_s,
                    tangential_velocity_m_s,
                    index,
                )?;
                loss_factor = prandtl_loss_factor(
                    self.n_blades,
                    station.radius_m,
                    self.hub_radius_m,
                    self.tip_radius_m,
                    section.inflow_angle_rad,
                    options.include_tip_loss,
                    options.include_root_loss,
                );
                let sin_phi_sq = section.inflow_angle_rad.sin().powi(2);
                let sin_cos_phi = section.inflow_angle_rad.sin() * section.inflow_angle_rad.cos();
                let target_axial_induction = if section.normal_force_coefficient > 0.0 {
                    let axial_denominator = 4.0 * loss_factor * sin_phi_sq
                        / (solidity * section.normal_force_coefficient)
                        - 1.0;
                    if !axial_denominator.is_finite() || axial_denominator <= 0.0 {
                        return Err(BladeElementError::InductionSingularity {
                            station: index,
                            equation: "axial",
                        });
                    }
                    1.0 / axial_denominator
                } else {
                    // A locally unloading or braking section does not receive
                    // a positive axial momentum induction from this reference.
                    0.0
                };
                let target_tangential_induction = if section.tangential_force_coefficient > 0.0 {
                    let tangential_denominator = 4.0 * loss_factor * sin_cos_phi
                        / (solidity * section.tangential_force_coefficient)
                        + 1.0;
                    if !tangential_denominator.is_finite() || tangential_denominator <= 0.0 {
                        return Err(BladeElementError::InductionSingularity {
                            station: index,
                            equation: "tangential",
                        });
                    }
                    1.0 / tangential_denominator
                } else {
                    0.0
                };
                validate_induction(index, "axial_induction", target_axial_induction)?;
                validate_induction(index, "tangential_induction", target_tangential_induction)?;
                let relaxed_axial = axial_induction
                    + options.relaxation * (target_axial_induction - axial_induction);
                let relaxed_tangential = tangential_induction
                    + options.relaxation * (target_tangential_induction - tangential_induction);
                residual = (relaxed_axial - axial_induction)
                    .abs()
                    .max((relaxed_tangential - tangential_induction).abs());
                axial_induction = relaxed_axial;
                tangential_induction = relaxed_tangential;
                iterations_used = iteration;
                if residual <= options.convergence_tolerance {
                    break;
                }
            }
            max_iterations_used = max_iterations_used.max(iterations_used);
            if residual > options.convergence_tolerance {
                return Err(BladeElementError::InductionNotConverged {
                    station: index,
                    iterations: iterations_used,
                    residual,
                });
            }

            let axial_velocity_m_s = operating_point.freestream_m_s * (1.0 + axial_induction)
                + external.axial_velocity_m_s;
            let tangential_velocity_m_s =
                omega_rad_s * station.radius_m * (1.0 - tangential_induction)
                    + external.tangential_velocity_m_s;
            let section = evaluate_section(
                &self.polar,
                station,
                operating_point,
                axial_velocity_m_s,
                tangential_velocity_m_s,
                index,
            )?;
            let strip_thrust_n = blade_count * section.axial_force_per_m_n * width_m;
            let strip_torque_nm =
                blade_count * section.tangential_force_per_m_n * station.radius_m * width_m;
            if !strip_thrust_n.is_finite() || !strip_torque_nm.is_finite() {
                return Err(BladeElementError::NonFiniteResult {
                    field: "bem_strip_load",
                });
            }
            thrust_n += strip_thrust_n;
            torque_nm += strip_torque_nm;
            loads.push(BladeElementStripLoad {
                radius_m: station.radius_m,
                width_m,
                chord_m: station.chord_m,
                pitch_deg: station.pitch_deg,
                inflow_angle_deg: section.inflow_angle_deg,
                angle_of_attack_deg: section.angle_of_attack_deg,
                relative_speed_m_s: section.relative_speed_m_s,
                lift_coefficient: section.lift_coefficient,
                drag_coefficient: section.drag_coefficient,
                axial_force_per_m_n: section.axial_force_per_m_n,
                tangential_force_per_m_n: section.tangential_force_per_m_n,
                thrust_n: strip_thrust_n,
                torque_nm: strip_torque_nm,
            });
            induction.push(BladeElementBemStrip {
                axial_induction,
                tangential_induction,
                loss_factor,
                iterations: iterations_used,
                residual,
            });
        }

        let shaft_power_w = torque_nm * omega_rad_s;
        let useful_power_w = thrust_n * operating_point.freestream_m_s;
        let propulsive_efficiency = if shaft_power_w > 0.0 {
            useful_power_w / shaft_power_w
        } else {
            0.0
        };
        for (field, value) in [
            ("bem.thrust_n", thrust_n),
            ("bem.torque_nm", torque_nm),
            ("bem.shaft_power_w", shaft_power_w),
            ("bem.useful_power_w", useful_power_w),
            ("bem.propulsive_efficiency", propulsive_efficiency),
        ] {
            if !value.is_finite() {
                return Err(BladeElementError::NonFiniteResult { field });
            }
        }
        Ok(BladeElementBemResult {
            loads: BladeElementResult {
                operating_point,
                omega_rad_s,
                advance_ratio,
                tip_speed_m_s,
                thrust_n,
                torque_nm,
                shaft_power_w,
                useful_power_w,
                propulsive_efficiency,
                strips: loads,
            },
            strips: induction,
            converged: true,
            max_iterations_used,
        })
    }

    /// Evaluate a deterministic ordered matrix of forward-flight BEM points.
    ///
    /// Every point is evaluated independently with the same geometry, polar,
    /// and solver options.  Results preserve input order so callers can join
    /// them to an external operating-condition table without relying on a
    /// sort or on floating-point ordering.  The method is intentionally a
    /// solver-free library boundary; it performs no artifact export or QA
    /// orchestration.
    pub fn evaluate_bem_multipoint(
        &self,
        operating_points: &[BladeElementOperatingPoint],
        options: BladeElementBemOptions,
    ) -> Result<Vec<BladeElementBemResult>, BladeElementError> {
        if operating_points.is_empty() {
            return Err(BladeElementError::InvalidInput {
                field: "operating_points",
                requirement: "non-empty",
            });
        }
        operating_points
            .iter()
            .copied()
            .enumerate()
            .map(|(point_index, operating_point)| {
                self.evaluate_bem(operating_point, options)
                    .map_err(|error| BladeElementError::MultipointEvaluationFailed {
                        point_index,
                        error: Box::new(error),
                    })
            })
            .collect()
    }

    /// Evaluate BEM loads on an explicit, strictly increasing station ladder.
    ///
    /// Each station vector is validated independently and evaluated with the
    /// same operating point, polar, and BEM options.  Relative changes use a
    /// symmetric scale with a small absolute floor, so a near-zero load cannot
    /// create an artificial divide-by-zero.  The method reports convergence of
    /// the integrated loads only; it does not claim temporal, wake, or CFD
    /// convergence.
    pub fn evaluate_bem_convergence(
        &self,
        station_ladders: &[Vec<BladeElementStation>],
        operating_point: BladeElementOperatingPoint,
        options: BladeElementBemOptions,
        relative_load_tolerance: f64,
    ) -> Result<BladeElementBemConvergence, BladeElementError> {
        if station_ladders.len() < 2 {
            return Err(BladeElementError::InvalidInput {
                field: "station_ladders",
                requirement: "at least two levels",
            });
        }
        if !relative_load_tolerance.is_finite() || relative_load_tolerance <= 0.0 {
            return Err(BladeElementError::InvalidInput {
                field: "relative_load_tolerance",
                requirement: "finite and positive",
            });
        }
        for (level_index, ladder) in station_ladders.iter().enumerate() {
            if ladder.is_empty() {
                return Err(BladeElementError::InvalidInput {
                    field: "station_ladders[level]",
                    requirement: "non-empty",
                });
            }
            if let Some(previous) = level_index
                .checked_sub(1)
                .and_then(|index| station_ladders.get(index))
                && ladder.len() <= previous.len()
            {
                return Err(BladeElementError::InvalidInput {
                    field: "station_ladders",
                    requirement: "strictly increasing station counts",
                });
            }
        }

        let mut levels = Vec::with_capacity(station_ladders.len());
        let mut results = Vec::with_capacity(station_ladders.len());
        let mut previous_thrust_n = None;
        let mut previous_torque_nm = None;
        for (level_index, stations) in station_ladders.iter().enumerate() {
            let candidate = self.with_stations(stations.clone()).map_err(|error| {
                BladeElementError::ConvergenceLevelFailed {
                    level_index,
                    error: Box::new(error),
                }
            })?;
            let result = candidate
                .evaluate_bem(operating_point, options)
                .map_err(|error| BladeElementError::ConvergenceLevelFailed {
                    level_index,
                    error: Box::new(error),
                })?;
            let relative_thrust_change =
                previous_thrust_n.map(|previous| relative_change(result.loads.thrust_n, previous));
            let relative_torque_change = previous_torque_nm
                .map(|previous| relative_change(result.loads.torque_nm, previous));
            let converged = relative_thrust_change
                .zip(relative_torque_change)
                .is_some_and(|(thrust, torque)| {
                    thrust <= relative_load_tolerance && torque <= relative_load_tolerance
                });
            levels.push(BladeElementBemConvergenceLevel {
                station_count: stations.len(),
                thrust_n: result.loads.thrust_n,
                torque_nm: result.loads.torque_nm,
                shaft_power_w: result.loads.shaft_power_w,
                relative_thrust_change,
                relative_torque_change,
                converged,
            });
            previous_thrust_n = Some(result.loads.thrust_n);
            previous_torque_nm = Some(result.loads.torque_nm);
            results.push(result);
        }
        let converged = levels.last().is_some_and(|level| level.converged);
        Ok(BladeElementBemConvergence {
            levels,
            results,
            converged,
        })
    }

    fn station_widths(&self) -> Vec<f64> {
        self.stations
            .iter()
            .enumerate()
            .map(|(index, station)| {
                let lower_boundary = if index == 0 {
                    self.hub_radius_m
                } else {
                    0.5 * (self.stations[index - 1].radius_m + station.radius_m)
                };
                let upper_boundary = if index + 1 == self.stations.len() {
                    self.tip_radius_m
                } else {
                    0.5 * (station.radius_m + self.stations[index + 1].radius_m)
                };
                upper_boundary - lower_boundary
            })
            .collect()
    }

    fn with_stations(&self, stations: Vec<BladeElementStation>) -> Result<Self, BladeElementError> {
        validate_station_layout(&stations, self.hub_radius_m, self.tip_radius_m)?;
        let mut candidate = self.clone();
        candidate.stations = stations;
        Ok(candidate)
    }
}

/// Compare one direct no-induction result with one converged BEM result.
///
/// The operating-point metadata must match exactly. Relative differences use
/// `max(|direct|, |bem|, relative_scale_floor)` and therefore remain defined
/// for zero or near-zero loads. This is numerical fidelity-ladder evidence
/// between two checked references; it is not CFD, free-wake, or measured-data
/// validation.
pub fn compare_direct_and_bem(
    direct: &BladeElementResult,
    bem: &BladeElementBemResult,
    options: BladeElementFidelityOptions,
) -> Result<BladeElementFidelityComparison, BladeElementFidelityError> {
    compare_direct_and_bem_at_index(direct, bem, options, 0)
}

fn compare_direct_and_bem_at_index(
    direct: &BladeElementResult,
    bem: &BladeElementBemResult,
    options: BladeElementFidelityOptions,
    index: usize,
) -> Result<BladeElementFidelityComparison, BladeElementFidelityError> {
    validate_fidelity_options(options)?;
    validate_fidelity_operating_point(direct.operating_point, "direct", index)?;
    validate_fidelity_operating_point(bem.loads.operating_point, "bem", index)?;
    if direct.operating_point != bem.loads.operating_point {
        return Err(BladeElementFidelityError::OperatingPointMismatch { index });
    }
    validate_bem_result(bem, index)?;
    validate_result_metrics(direct, "direct", index)?;
    validate_result_metrics(&bem.loads, "bem", index)?;

    let relative_thrust_difference = relative_metric_difference(
        "thrust",
        direct.thrust_n,
        bem.loads.thrust_n,
        options.relative_scale_floor,
    )?;
    let relative_torque_difference = relative_metric_difference(
        "torque",
        direct.torque_nm,
        bem.loads.torque_nm,
        options.relative_scale_floor,
    )?;
    let relative_shaft_power_difference = relative_metric_difference(
        "shaft_power",
        direct.shaft_power_w,
        bem.loads.shaft_power_w,
        options.relative_scale_floor,
    )?;
    let relative_propulsive_efficiency_difference = relative_metric_difference(
        "propulsive_efficiency",
        direct.propulsive_efficiency,
        bem.loads.propulsive_efficiency,
        options.relative_scale_floor,
    )?;

    Ok(BladeElementFidelityComparison {
        operating_point: direct.operating_point,
        direct_thrust_n: direct.thrust_n,
        bem_thrust_n: bem.loads.thrust_n,
        relative_thrust_difference,
        direct_torque_nm: direct.torque_nm,
        bem_torque_nm: bem.loads.torque_nm,
        relative_torque_difference,
        direct_shaft_power_w: direct.shaft_power_w,
        bem_shaft_power_w: bem.loads.shaft_power_w,
        relative_shaft_power_difference,
        direct_propulsive_efficiency: direct.propulsive_efficiency,
        bem_propulsive_efficiency: bem.loads.propulsive_efficiency,
        relative_propulsive_efficiency_difference,
    })
}

/// Compare ordered direct/BEM result matrices without sorting away point
/// association. The returned vector has exactly the caller's input order.
pub fn compare_direct_and_bem_multipoint(
    direct: &[BladeElementResult],
    bem: &[BladeElementBemResult],
    options: BladeElementFidelityOptions,
) -> Result<Vec<BladeElementFidelityComparison>, BladeElementFidelityError> {
    validate_fidelity_options(options)?;
    if direct.len() != bem.len() {
        return Err(BladeElementFidelityError::ResultCountMismatch {
            direct_count: direct.len(),
            bem_count: bem.len(),
        });
    }
    if direct.is_empty() {
        return Err(BladeElementFidelityError::EmptyResults {
            field: "multipoint_results",
        });
    }
    direct
        .iter()
        .zip(bem)
        .enumerate()
        .map(|(index, (direct, bem))| compare_direct_and_bem_at_index(direct, bem, options, index))
        .collect()
}

/// Check whether direct and BEM candidate rankings agree at one operating
/// point.
///
/// Candidate indices are retained rather than returning a single boolean, so
/// a caller can audit exactly which candidate moved. Actual rank changes are
/// valid evidence and return `preserved == false`; malformed sets,
/// non-converged BEM results, non-finite metrics, and ambiguous ties fail
/// closed. The result is a screening comparison, not a claim that either
/// model is the physical truth.
pub fn assess_direct_bem_rank_preservation(
    direct: &[BladeElementResult],
    bem: &[BladeElementBemResult],
    metric: BladeElementRankMetric,
    order: BladeElementRankOrder,
    options: BladeElementFidelityOptions,
) -> Result<BladeElementRankEvidence, BladeElementFidelityError> {
    validate_fidelity_options(options)?;
    if direct.len() != bem.len() {
        return Err(BladeElementFidelityError::ResultCountMismatch {
            direct_count: direct.len(),
            bem_count: bem.len(),
        });
    }
    if direct.is_empty() {
        return Err(BladeElementFidelityError::EmptyResults {
            field: "rank_candidates",
        });
    }
    if direct.len() < 2 {
        return Err(BladeElementFidelityError::InsufficientCandidates {
            count: direct.len(),
        });
    }

    let direct_point = direct[0].operating_point;
    let bem_point = bem[0].loads.operating_point;
    let mut direct_values = Vec::with_capacity(direct.len());
    let mut bem_values = Vec::with_capacity(bem.len());
    for (index, (direct_result, bem_result)) in direct.iter().zip(bem).enumerate() {
        compare_direct_and_bem(direct_result, bem_result, options)?;
        if direct_result.operating_point != direct_point
            || bem_result.loads.operating_point != bem_point
        {
            return Err(BladeElementFidelityError::CandidateOperatingPointMismatch { index });
        }
        if direct_point != bem_point {
            return Err(BladeElementFidelityError::OperatingPointMismatch { index });
        }
        direct_values.push(metric_value(direct_result, metric));
        bem_values.push(metric_value(&bem_result.loads, metric));
    }

    let direct_candidate_order = rank_values(&direct_values, "direct", metric, order, options)?;
    let bem_candidate_order = rank_values(&bem_values, "bem", metric, order, options)?;
    let preserved = direct_candidate_order == bem_candidate_order;

    Ok(BladeElementRankEvidence {
        operating_point: direct_point,
        metric,
        order,
        direct_candidate_order,
        bem_candidate_order,
        preserved,
    })
}

fn validate_fidelity_options(
    options: BladeElementFidelityOptions,
) -> Result<(), BladeElementFidelityError> {
    if !options.relative_scale_floor.is_finite() || options.relative_scale_floor <= 0.0 {
        return Err(BladeElementFidelityError::InvalidOptions {
            field: "relative_scale_floor",
            requirement: "finite and positive",
        });
    }
    if !options.rank_tie_tolerance.is_finite() || options.rank_tie_tolerance < 0.0 {
        return Err(BladeElementFidelityError::InvalidOptions {
            field: "rank_tie_tolerance",
            requirement: "finite and non-negative",
        });
    }
    Ok(())
}

fn validate_fidelity_operating_point(
    point: BladeElementOperatingPoint,
    model: &'static str,
    index: usize,
) -> Result<(), BladeElementFidelityError> {
    if !point.rpm.is_finite()
        || point.rpm <= 0.0
        || !point.freestream_m_s.is_finite()
        || point.freestream_m_s < 0.0
        || !point.density_kg_m3.is_finite()
        || point.density_kg_m3 <= 0.0
    {
        return Err(BladeElementFidelityError::InvalidOperatingPoint { model, index });
    }
    Ok(())
}

fn validate_result_metrics(
    result: &BladeElementResult,
    model: &'static str,
    index: usize,
) -> Result<(), BladeElementFidelityError> {
    for (metric, value) in [
        (BladeElementRankMetric::Thrust, result.thrust_n),
        (BladeElementRankMetric::Torque, result.torque_nm),
        (BladeElementRankMetric::ShaftPower, result.shaft_power_w),
        (
            BladeElementRankMetric::PropulsiveEfficiency,
            result.propulsive_efficiency,
        ),
    ] {
        if !value.is_finite() {
            return Err(BladeElementFidelityError::NonFiniteMetric {
                model,
                index,
                metric,
            });
        }
    }
    Ok(())
}

fn validate_bem_result(
    result: &BladeElementBemResult,
    index: usize,
) -> Result<(), BladeElementFidelityError> {
    if !result.converged || result.max_iterations_used == 0 {
        return Err(BladeElementFidelityError::UnconvergedBem { index });
    }
    if result.strips.len() != result.loads.strips.len() {
        return Err(BladeElementFidelityError::UnconvergedBem { index });
    }
    Ok(())
}

fn relative_metric_difference(
    field: &'static str,
    direct: f64,
    bem: f64,
    scale_floor: f64,
) -> Result<f64, BladeElementFidelityError> {
    let difference = (direct - bem).abs();
    let scale = direct.abs().max(bem.abs()).max(scale_floor);
    let relative_difference = difference / scale;
    if !relative_difference.is_finite() {
        return Err(BladeElementFidelityError::NonFiniteDifference { field });
    }
    Ok(relative_difference)
}

fn metric_value(result: &BladeElementResult, metric: BladeElementRankMetric) -> f64 {
    match metric {
        BladeElementRankMetric::Thrust => result.thrust_n,
        BladeElementRankMetric::Torque => result.torque_nm,
        BladeElementRankMetric::ShaftPower => result.shaft_power_w,
        BladeElementRankMetric::PropulsiveEfficiency => result.propulsive_efficiency,
    }
}

fn rank_values(
    values: &[f64],
    model: &'static str,
    metric: BladeElementRankMetric,
    order: BladeElementRankOrder,
    options: BladeElementFidelityOptions,
) -> Result<Vec<usize>, BladeElementFidelityError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(BladeElementFidelityError::NonFiniteMetric {
                model,
                index,
                metric,
            });
        }
    }
    for first_candidate in 0..values.len() {
        for second_candidate in (first_candidate + 1)..values.len() {
            let relative_difference = relative_metric_difference(
                "rank",
                values[first_candidate],
                values[second_candidate],
                options.relative_scale_floor,
            )?;
            if relative_difference <= options.rank_tie_tolerance {
                return Err(BladeElementFidelityError::AmbiguousRank {
                    model,
                    first_candidate,
                    second_candidate,
                    metric,
                    relative_difference,
                    tolerance: options.rank_tie_tolerance,
                });
            }
        }
    }

    let mut candidate_order = (0..values.len()).collect::<Vec<_>>();
    candidate_order.sort_by(|left, right| {
        let comparison = values[*left]
            .partial_cmp(&values[*right])
            .expect("rank_values validates finite values");
        match order {
            BladeElementRankOrder::Ascending => comparison,
            BladeElementRankOrder::Descending => comparison.reverse(),
        }
    });
    Ok(candidate_order)
}

#[derive(Debug, Clone, Copy)]
struct SectionEvaluation {
    inflow_angle_rad: f64,
    inflow_angle_deg: f64,
    angle_of_attack_deg: f64,
    relative_speed_m_s: f64,
    lift_coefficient: f64,
    drag_coefficient: f64,
    axial_force_per_m_n: f64,
    tangential_force_per_m_n: f64,
    normal_force_coefficient: f64,
    tangential_force_coefficient: f64,
}

fn evaluate_section(
    polar: &AirfoilPolar,
    station: &BladeElementStation,
    operating_point: BladeElementOperatingPoint,
    axial_velocity_m_s: f64,
    tangential_velocity_m_s: f64,
    station_index: usize,
) -> Result<SectionEvaluation, BladeElementError> {
    if !axial_velocity_m_s.is_finite() || axial_velocity_m_s < 0.0 {
        return Err(BladeElementError::InvalidInput {
            field: "axial_velocity_m_s",
            requirement: "finite and non-negative",
        });
    }
    if !tangential_velocity_m_s.is_finite() || tangential_velocity_m_s <= 0.0 {
        return Err(BladeElementError::InvalidInput {
            field: "tangential_velocity_m_s",
            requirement: "finite and positive",
        });
    }
    let relative_speed_m_s = axial_velocity_m_s.hypot(tangential_velocity_m_s);
    let inflow_angle_rad = axial_velocity_m_s.atan2(tangential_velocity_m_s);
    let inflow_angle_deg = inflow_angle_rad.to_degrees();
    let angle_of_attack_deg = station.pitch_deg - inflow_angle_deg;
    let (lift_coefficient, drag_coefficient) =
        polar
            .coefficients(angle_of_attack_deg)
            .map_err(|error| match error {
                BladeElementError::PolarQueryOutsideRange {
                    angle_of_attack_deg,
                    minimum_deg,
                    maximum_deg,
                } => BladeElementError::AngleOfAttackOutsidePolar {
                    station: station_index,
                    angle_of_attack_deg,
                    minimum_deg,
                    maximum_deg,
                },
                other => other,
            })?;
    let dynamic_pressure_pa =
        0.5 * operating_point.density_kg_m3 * relative_speed_m_s * relative_speed_m_s;
    let lift_per_m_n = dynamic_pressure_pa * station.chord_m * lift_coefficient;
    let drag_per_m_n = dynamic_pressure_pa * station.chord_m * drag_coefficient;
    let (sin_phi, cos_phi) = inflow_angle_rad.sin_cos();
    Ok(SectionEvaluation {
        inflow_angle_rad,
        inflow_angle_deg,
        angle_of_attack_deg,
        relative_speed_m_s,
        lift_coefficient,
        drag_coefficient,
        axial_force_per_m_n: lift_per_m_n * cos_phi - drag_per_m_n * sin_phi,
        tangential_force_per_m_n: lift_per_m_n * sin_phi + drag_per_m_n * cos_phi,
        normal_force_coefficient: lift_coefficient * cos_phi - drag_coefficient * sin_phi,
        tangential_force_coefficient: lift_coefficient * sin_phi + drag_coefficient * cos_phi,
    })
}

fn prandtl_loss_factor(
    n_blades: usize,
    radius_m: f64,
    hub_radius_m: f64,
    tip_radius_m: f64,
    inflow_angle_rad: f64,
    include_tip_loss: bool,
    include_root_loss: bool,
) -> f64 {
    let sin_phi = inflow_angle_rad.sin().abs().max(1.0e-8);
    let blade_factor = 0.5 * n_blades as f64;
    let tip_exponent = blade_factor * (tip_radius_m - radius_m) / (radius_m * sin_phi);
    let tip_factor = if include_tip_loss {
        (2.0 / PI) * (-tip_exponent.max(0.0)).exp().clamp(0.0, 1.0).acos()
    } else {
        1.0
    };
    let root_factor = if include_root_loss && hub_radius_m > 0.0 {
        let root_exponent = blade_factor * (radius_m - hub_radius_m) / (hub_radius_m * sin_phi);
        (2.0 / PI) * (-root_exponent.max(0.0)).exp().clamp(0.0, 1.0).acos()
    } else {
        1.0
    };
    (tip_factor * root_factor).clamp(1.0e-3, 1.0)
}

fn validate_induction(
    station: usize,
    field: &'static str,
    value: f64,
) -> Result<(), BladeElementError> {
    if !value.is_finite() || !(0.0..=0.95).contains(&value) {
        return Err(BladeElementError::InductionOutOfRange { station, field });
    }
    Ok(())
}

fn validate_bem_options(options: BladeElementBemOptions) -> Result<(), BladeElementError> {
    if options.max_iterations == 0 {
        return Err(BladeElementError::InvalidBemOptions {
            field: "max_iterations",
            requirement: "non-zero",
        });
    }
    if !options.convergence_tolerance.is_finite() || options.convergence_tolerance <= 0.0 {
        return Err(BladeElementError::InvalidBemOptions {
            field: "convergence_tolerance",
            requirement: "finite and positive",
        });
    }
    if !options.relaxation.is_finite()
        || !(0.0..=1.0).contains(&options.relaxation)
        || options.relaxation == 0.0
    {
        return Err(BladeElementError::InvalidBemOptions {
            field: "relaxation",
            requirement: "finite in (0, 1]",
        });
    }
    Ok(())
}

fn validate_station_layout(
    stations: &[BladeElementStation],
    hub_radius_m: f64,
    tip_radius_m: f64,
) -> Result<(), BladeElementError> {
    if stations.is_empty() {
        return Err(BladeElementError::InvalidInput {
            field: "stations",
            requirement: "non-empty",
        });
    }
    for (index, station) in stations.iter().enumerate() {
        if !station.radius_m.is_finite() {
            return Err(BladeElementError::InvalidStation {
                index,
                field: "radius_m",
                requirement: "finite",
            });
        }
        if station.radius_m <= hub_radius_m || station.radius_m >= tip_radius_m {
            return Err(BladeElementError::InvalidStation {
                index,
                field: "radius_m",
                requirement: "strictly inside hub/tip radii",
            });
        }
        if !station.chord_m.is_finite() || station.chord_m <= 0.0 {
            return Err(BladeElementError::InvalidStation {
                index,
                field: "chord_m",
                requirement: "finite and positive",
            });
        }
        if !station.pitch_deg.is_finite() {
            return Err(BladeElementError::InvalidStation {
                index,
                field: "pitch_deg",
                requirement: "finite",
            });
        }
        if let Some(previous) = index.checked_sub(1).and_then(|i| stations.get(i))
            && station.radius_m <= previous.radius_m
        {
            return Err(BladeElementError::NonIncreasingStationRadius { index });
        }
    }
    Ok(())
}

fn relative_change(current: f64, previous: f64) -> f64 {
    (current - previous).abs() / current.abs().max(previous.abs()).max(1.0e-12)
}

fn validate_operating_point(
    operating_point: BladeElementOperatingPoint,
) -> Result<(), BladeElementError> {
    for (field, value) in [
        ("operating_point.rpm", operating_point.rpm),
        (
            "operating_point.freestream_m_s",
            operating_point.freestream_m_s,
        ),
        (
            "operating_point.density_kg_m3",
            operating_point.density_kg_m3,
        ),
    ] {
        if !value.is_finite() {
            return Err(BladeElementError::InvalidInput {
                field,
                requirement: "finite",
            });
        }
    }
    if operating_point.rpm <= 0.0 {
        return Err(BladeElementError::InvalidInput {
            field: "operating_point.rpm",
            requirement: "positive",
        });
    }
    if operating_point.freestream_m_s < 0.0 {
        return Err(BladeElementError::InvalidInput {
            field: "operating_point.freestream_m_s",
            requirement: "non-negative",
        });
    }
    if operating_point.density_kg_m3 <= 0.0 {
        return Err(BladeElementError::InvalidInput {
            field: "operating_point.density_kg_m3",
            requirement: "positive",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fidelity_pair() -> (BladeElementResult, BladeElementBemResult) {
        let model = model();
        let operating_point = BladeElementOperatingPoint {
            rpm: 12_000.0,
            freestream_m_s: 10.0,
            density_kg_m3: 1.204,
        };
        let direct = model.evaluate(operating_point).expect("direct result");
        let bem = model
            .evaluate_bem(operating_point, BladeElementBemOptions::default())
            .expect("BEM result");
        (direct, bem)
    }

    #[test]
    fn fidelity_comparison_preserves_point_and_reports_finite_differences() {
        let (direct, mut bem) = fidelity_pair();
        bem.loads = direct.clone();
        let comparison =
            compare_direct_and_bem(&direct, &bem, BladeElementFidelityOptions::default())
                .expect("identical reference metrics");
        assert_eq!(comparison.operating_point, direct.operating_point);
        assert_eq!(comparison.relative_thrust_difference, 0.0);
        assert_eq!(comparison.relative_torque_difference, 0.0);
        assert_eq!(comparison.relative_shaft_power_difference, 0.0);
        assert_eq!(comparison.relative_propulsive_efficiency_difference, 0.0);

        let (_, bem_reference) = fidelity_pair();
        let comparison = compare_direct_and_bem(
            &direct,
            &bem_reference,
            BladeElementFidelityOptions::default(),
        )
        .expect("finite direct/BEM comparison");
        assert!(comparison.relative_thrust_difference.is_finite());
        assert!(comparison.relative_torque_difference.is_finite());
        assert!(comparison.relative_shaft_power_difference.is_finite());
        assert!(
            comparison
                .relative_propulsive_efficiency_difference
                .is_finite()
        );
    }

    #[test]
    fn fidelity_comparison_uses_floor_and_rejects_bad_metadata() {
        let (mut direct, mut bem) = fidelity_pair();
        direct.thrust_n = 0.0;
        direct.torque_nm = 0.0;
        direct.shaft_power_w = 0.0;
        direct.propulsive_efficiency = 0.0;
        bem.loads = direct.clone();
        bem.loads.thrust_n = 1.0e-9;
        let options = BladeElementFidelityOptions {
            relative_scale_floor: 1.0e-3,
            ..BladeElementFidelityOptions::default()
        };
        let comparison = compare_direct_and_bem(&direct, &bem, options)
            .expect("floor keeps near-zero comparison defined");
        assert!((comparison.relative_thrust_difference - 1.0e-6).abs() < 1.0e-15);

        let mut invalid_options = options;
        invalid_options.relative_scale_floor = 0.0;
        assert!(matches!(
            compare_direct_and_bem(&direct, &bem, invalid_options),
            Err(BladeElementFidelityError::InvalidOptions {
                field: "relative_scale_floor",
                ..
            })
        ));

        let mut invalid_direct = direct.clone();
        invalid_direct.thrust_n = f64::NAN;
        assert!(matches!(
            compare_direct_and_bem(&invalid_direct, &bem, options),
            Err(BladeElementFidelityError::NonFiniteMetric {
                model: "direct",
                metric: BladeElementRankMetric::Thrust,
                ..
            })
        ));

        let mut mismatched_bem = bem.clone();
        mismatched_bem.loads.operating_point.freestream_m_s += 1.0;
        assert!(matches!(
            compare_direct_and_bem(&direct, &mismatched_bem, options),
            Err(BladeElementFidelityError::OperatingPointMismatch { index: 0 })
        ));

        let mut unconverged_bem = bem;
        unconverged_bem.converged = false;
        assert!(matches!(
            compare_direct_and_bem(&direct, &unconverged_bem, options),
            Err(BladeElementFidelityError::UnconvergedBem { index: 0 })
        ));
    }

    #[test]
    fn fidelity_multipoint_comparison_is_ordered_and_cardinality_checked() {
        let model = model();
        let points = [
            BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 10.0,
                density_kg_m3: 1.204,
            },
            BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 14.0,
                density_kg_m3: 1.204,
            },
        ];
        let direct = points
            .iter()
            .map(|point| model.evaluate(*point).expect("direct point"))
            .collect::<Vec<_>>();
        let bem = points
            .iter()
            .map(|point| {
                model
                    .evaluate_bem(*point, BladeElementBemOptions::default())
                    .expect("BEM point")
            })
            .collect::<Vec<_>>();
        let comparisons = compare_direct_and_bem_multipoint(
            &direct,
            &bem,
            BladeElementFidelityOptions::default(),
        )
        .expect("ordered multipoint comparison");
        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.operating_point)
                .collect::<Vec<_>>(),
            points
        );
        assert!(
            comparisons
                .iter()
                .all(|comparison| comparison.relative_thrust_difference.is_finite())
        );

        let mut mismatched_bem = bem.clone();
        mismatched_bem[1].loads.operating_point.freestream_m_s += 1.0;
        assert!(matches!(
            compare_direct_and_bem_multipoint(
                &direct,
                &mismatched_bem,
                BladeElementFidelityOptions::default(),
            ),
            Err(BladeElementFidelityError::OperatingPointMismatch { index: 1 })
        ));

        assert!(matches!(
            compare_direct_and_bem_multipoint(
                &direct,
                &bem[..1],
                BladeElementFidelityOptions::default(),
            ),
            Err(BladeElementFidelityError::ResultCountMismatch {
                direct_count: 2,
                bem_count: 1,
            })
        ));
        assert!(matches!(
            compare_direct_and_bem_multipoint(&[], &[], BladeElementFidelityOptions::default(),),
            Err(BladeElementFidelityError::EmptyResults { .. })
        ));
    }

    #[test]
    fn fidelity_rank_evidence_reports_preservation_changes_and_ties() {
        let (base_direct, base_bem) = fidelity_pair();
        let mut direct_first = base_direct.clone();
        let mut direct_second = base_direct;
        direct_first.thrust_n = 10.0;
        direct_second.thrust_n = 20.0;
        let mut bem_first = base_bem.clone();
        let mut bem_second = base_bem.clone();
        bem_first.loads.thrust_n = 12.0;
        bem_second.loads.thrust_n = 24.0;

        let direct = [direct_first.clone(), direct_second.clone()];
        let bem = [bem_first.clone(), bem_second.clone()];
        let evidence = assess_direct_bem_rank_preservation(
            &direct,
            &bem,
            BladeElementRankMetric::Thrust,
            BladeElementRankOrder::Descending,
            BladeElementFidelityOptions::default(),
        )
        .expect("strict preserved ranking");
        assert_eq!(evidence.direct_candidate_order, vec![1, 0]);
        assert_eq!(evidence.bem_candidate_order, vec![1, 0]);
        assert!(evidence.preserved);

        bem_first.loads.thrust_n = 30.0;
        let changed = assess_direct_bem_rank_preservation(
            &direct,
            &[bem_first.clone(), bem_second.clone()],
            BladeElementRankMetric::Thrust,
            BladeElementRankOrder::Descending,
            BladeElementFidelityOptions::default(),
        )
        .expect("rank change is evidence, not malformed input");
        assert_eq!(changed.bem_candidate_order, vec![0, 1]);
        assert!(!changed.preserved);

        let tied = [direct_first.clone(), direct_first.clone()];
        assert!(matches!(
            assess_direct_bem_rank_preservation(
                &tied,
                &bem,
                BladeElementRankMetric::Thrust,
                BladeElementRankOrder::Descending,
                BladeElementFidelityOptions::default(),
            ),
            Err(BladeElementFidelityError::AmbiguousRank {
                model: "direct",
                metric: BladeElementRankMetric::Thrust,
                ..
            })
        ));

        let mut second_point = direct_second;
        let mut bem_second_point = bem_second;
        second_point.operating_point.freestream_m_s += 1.0;
        bem_second_point.loads.operating_point.freestream_m_s += 1.0;
        assert!(matches!(
            assess_direct_bem_rank_preservation(
                &[direct_first, second_point],
                &[bem_first, bem_second_point],
                BladeElementRankMetric::Thrust,
                BladeElementRankOrder::Descending,
                BladeElementFidelityOptions::default(),
            ),
            Err(BladeElementFidelityError::CandidateOperatingPointMismatch { index: 1 })
        ));
    }

    fn source() -> PropellerSource {
        PropellerSource::drone_5inch(12_000.0)
    }

    fn polar() -> AirfoilPolar {
        AirfoilPolar::try_new(vec![
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
        .expect("test polar is valid")
    }

    fn stations(pitch_deg: f64) -> Vec<BladeElementStation> {
        vec![
            BladeElementStation {
                radius_m: 0.018,
                chord_m: 0.018,
                pitch_deg,
            },
            BladeElementStation {
                radius_m: 0.036,
                chord_m: 0.014,
                pitch_deg: pitch_deg - 3.0,
            },
            BladeElementStation {
                radius_m: 0.054,
                chord_m: 0.010,
                pitch_deg: pitch_deg - 6.0,
            },
        ]
    }

    fn refinement_stations(count: usize) -> Vec<BladeElementStation> {
        let source = source();
        (1..=count)
            .map(|index| {
                let fraction = index as f64 / (count + 1) as f64;
                BladeElementStation {
                    radius_m: source.hub_radius
                        + fraction * (source.tip_radius - source.hub_radius),
                    chord_m: 0.018 - 0.008 * fraction,
                    pitch_deg: 32.0 - 8.0 * fraction,
                }
            })
            .collect()
    }

    fn model() -> BladeElementModel {
        BladeElementModel::try_from_source(&source(), stations(24.0), polar())
            .expect("test blade-element model is valid")
    }

    #[test]
    fn polar_interpolates_and_rejects_extrapolation() {
        let coefficients = polar().coefficients(10.0).expect("in-range angle");
        assert!((coefficients.0 - 0.4).abs() < 1.0e-12);
        assert!((coefficients.1 - 0.05).abs() < 1.0e-12);
        assert!(matches!(
            polar().coefficients(41.0),
            Err(BladeElementError::PolarQueryOutsideRange { .. })
        ));
    }

    #[test]
    fn hover_result_closes_force_and_shaft_power_ledger() {
        let result = model()
            .evaluate_at_freestream(0.0)
            .expect("hover is inside the polar");
        assert!(result.thrust_n > 0.0);
        assert!(result.torque_nm > 0.0);
        assert!((result.shaft_power_w - result.torque_nm * result.omega_rad_s).abs() < 1.0e-10);
        assert_eq!(result.useful_power_w, 0.0);
        assert_eq!(result.propulsive_efficiency, 0.0);
        let total_width: f64 = result.strips.iter().map(|strip| strip.width_m).sum();
        assert!((total_width - (source().tip_radius - source().hub_radius)).abs() < 1.0e-12);
    }

    #[test]
    fn explicit_density_and_rpm_scaling_matches_quadratic_blade_element_law() {
        let model = model();
        let base = model
            .evaluate(BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 0.0,
                density_kg_m3: 1.0,
            })
            .expect("base point");
        let doubled_density = model
            .evaluate(BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 0.0,
                density_kg_m3: 2.0,
            })
            .expect("density point");
        let doubled_rpm = model
            .evaluate(BladeElementOperatingPoint {
                rpm: 24_000.0,
                freestream_m_s: 0.0,
                density_kg_m3: 1.0,
            })
            .expect("rpm point");
        assert!((doubled_density.thrust_n / base.thrust_n - 2.0).abs() < 1.0e-12);
        assert!((doubled_density.torque_nm / base.torque_nm - 2.0).abs() < 1.0e-12);
        assert!((doubled_rpm.thrust_n / base.thrust_n - 4.0).abs() < 1.0e-12);
        assert!((doubled_rpm.torque_nm / base.torque_nm - 4.0).abs() < 1.0e-12);
        assert!((doubled_rpm.shaft_power_w / base.shaft_power_w - 8.0).abs() < 1.0e-12);
    }

    #[test]
    fn forward_inflow_changes_angle_of_attack_and_load() {
        let hover = model().evaluate_at_freestream(0.0).expect("hover point");
        let forward = model().evaluate_at_freestream(10.0).expect("forward point");
        assert!(forward.strips[0].inflow_angle_deg > hover.strips[0].inflow_angle_deg);
        assert!(forward.strips[0].angle_of_attack_deg < hover.strips[0].angle_of_attack_deg);
        assert!(forward.thrust_n < hover.thrust_n);
        assert!(forward.advance_ratio > 0.0);
        assert!(forward.propulsive_efficiency > 0.0);
    }

    #[test]
    fn bem_forward_flight_converges_and_closes_its_power_ledger() {
        let operating_point = BladeElementOperatingPoint {
            rpm: 12_000.0,
            freestream_m_s: 10.0,
            density_kg_m3: 1.204,
        };
        let direct = model()
            .evaluate(operating_point)
            .expect("direct forward-flight reference");
        let bem = model()
            .evaluate_bem(operating_point, BladeElementBemOptions::default())
            .expect("BEM forward-flight reference");
        assert!(bem.converged);
        assert_eq!(bem.strips.len(), model().stations().len());
        assert!(bem.max_iterations_used > 0);
        assert!(
            (bem.loads.shaft_power_w - bem.loads.torque_nm * bem.loads.omega_rad_s).abs() < 1.0e-10
        );
        assert!(bem.loads.thrust_n.is_finite());
        assert!(bem.loads.torque_nm.is_finite());
        assert!((bem.loads.thrust_n - direct.thrust_n).abs() > 1.0e-8);
        for strip in &bem.strips {
            assert!((0.0..=0.95).contains(&strip.axial_induction));
            assert!((0.0..=0.95).contains(&strip.tangential_induction));
            assert!((1.0e-3..=1.0).contains(&strip.loss_factor));
            assert!(strip.residual <= BladeElementBemOptions::default().convergence_tolerance);
        }
    }

    #[test]
    fn bem_loss_switch_and_invalid_boundaries_are_explicit() {
        let point = BladeElementOperatingPoint {
            rpm: 12_000.0,
            freestream_m_s: 14.0,
            density_kg_m3: 1.204,
        };
        let no_loss = BladeElementBemOptions {
            include_tip_loss: false,
            include_root_loss: false,
            ..BladeElementBemOptions::default()
        };
        let result = model()
            .evaluate_bem(point, no_loss)
            .expect("BEM without finite-blade losses");
        assert!(result.strips.iter().all(|strip| strip.loss_factor == 1.0));
        assert!(matches!(
            model().evaluate_bem(
                BladeElementOperatingPoint {
                    freestream_m_s: 0.0,
                    ..point
                },
                BladeElementBemOptions::default(),
            ),
            Err(BladeElementError::BemRequiresPositiveFreestream)
        ));
        assert!(matches!(
            model().evaluate_bem(
                point,
                BladeElementBemOptions {
                    max_iterations: 0,
                    ..BladeElementBemOptions::default()
                },
            ),
            Err(BladeElementError::InvalidBemOptions { .. })
        ));
        assert!(matches!(
            model().evaluate_bem(
                point,
                BladeElementBemOptions {
                    max_iterations: 1,
                    ..BladeElementBemOptions::default()
                },
            ),
            Err(BladeElementError::InductionNotConverged { .. })
        ));
    }

    #[test]
    fn bem_axial_momentum_balance_matches_blade_element_thrust() {
        let model = model();
        let point = BladeElementOperatingPoint {
            rpm: 12_000.0,
            freestream_m_s: 10.0,
            density_kg_m3: 1.204,
        };
        let result = model
            .evaluate_bem(point, BladeElementBemOptions::default())
            .expect("BEM reference");
        for (load, induction) in result.loads.strips.iter().zip(result.strips.iter()) {
            if induction.axial_induction <= 1.0e-8 {
                continue;
            }
            let momentum_thrust_n = 4.0
                * PI
                * point.density_kg_m3
                * point.freestream_m_s.powi(2)
                * induction.axial_induction
                * (1.0 + induction.axial_induction)
                * induction.loss_factor
                * load.radius_m
                * load.width_m;
            let scale = load.thrust_n.abs().max(momentum_thrust_n.abs()).max(1.0);
            assert!((load.thrust_n - momentum_thrust_n).abs() / scale < 1.0e-7);
        }
    }

    #[test]
    fn bem_multipoint_evaluation_is_ordered_deterministic_and_fail_closed() {
        let model = model();
        let points = [
            BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 10.0,
                density_kg_m3: 1.204,
            },
            BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 12.0,
                density_kg_m3: 1.18,
            },
            BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 14.0,
                density_kg_m3: 1.25,
            },
        ];
        let first = model
            .evaluate_bem_multipoint(&points, BladeElementBemOptions::default())
            .expect("valid multipoint matrix");
        let second = model
            .evaluate_bem_multipoint(&points, BladeElementBemOptions::default())
            .expect("deterministic replay");
        assert_eq!(first, second);
        assert_eq!(first.len(), points.len());
        for (result, point) in first.iter().zip(points) {
            assert_eq!(result.loads.operating_point, point);
            assert!(result.converged);
        }
        assert!(matches!(
            model.evaluate_bem_multipoint(&[], BladeElementBemOptions::default()),
            Err(BladeElementError::InvalidInput {
                field: "operating_points",
                ..
            })
        ));
        let invalid = [
            points[0],
            BladeElementOperatingPoint {
                rpm: 0.0,
                ..points[1]
            },
        ];
        assert!(matches!(
            model.evaluate_bem_multipoint(&invalid, BladeElementBemOptions::default()),
            Err(BladeElementError::MultipointEvaluationFailed { point_index: 1, .. })
        ));
    }

    #[test]
    fn bem_station_refinement_reports_deterministic_load_convergence() {
        let model = model();
        let ladders = vec![
            refinement_stations(4),
            refinement_stations(8),
            refinement_stations(16),
        ];
        let point = BladeElementOperatingPoint {
            rpm: 12_000.0,
            freestream_m_s: 14.0,
            density_kg_m3: 1.204,
        };
        let report = model
            .evaluate_bem_convergence(&ladders, point, BladeElementBemOptions::default(), 0.05)
            .expect("valid radial refinement ladder");
        assert!(report.converged);
        assert_eq!(report.levels.len(), 3);
        assert_eq!(report.results.len(), 3);
        assert_eq!(report.levels[0].station_count, 4);
        assert!(report.levels[0].relative_thrust_change.is_none());
        assert!(report.levels[0].relative_torque_change.is_none());
        for (level, result) in report.levels.iter().zip(report.results.iter()) {
            assert_eq!(level.station_count, result.loads.strips.len());
            assert!(level.thrust_n.is_finite());
            assert!(level.torque_nm.is_finite());
            assert!(level.shaft_power_w.is_finite());
            assert!(result.converged);
        }
        assert!(report.levels[2].converged);
        assert!(
            report.levels[2]
                .relative_thrust_change
                .expect("final thrust change")
                <= 0.05
        );
        assert!(
            report.levels[2]
                .relative_torque_change
                .expect("final torque change")
                <= 0.05
        );
        let replay = model
            .evaluate_bem_convergence(&ladders, point, BladeElementBemOptions::default(), 0.05)
            .expect("deterministic replay");
        assert_eq!(report, replay);
    }

    #[test]
    fn bem_station_refinement_rejects_malformed_ladders_with_context() {
        let model = model();
        let point = BladeElementOperatingPoint {
            rpm: 12_000.0,
            freestream_m_s: 14.0,
            density_kg_m3: 1.204,
        };
        let valid = refinement_stations(4);
        assert!(matches!(
            model.evaluate_bem_convergence(
                std::slice::from_ref(&valid),
                point,
                BladeElementBemOptions::default(),
                0.05,
            ),
            Err(BladeElementError::InvalidInput {
                field: "station_ladders",
                ..
            })
        ));
        assert!(matches!(
            model.evaluate_bem_convergence(
                &[refinement_stations(4), refinement_stations(4)],
                point,
                BladeElementBemOptions::default(),
                0.05,
            ),
            Err(BladeElementError::InvalidInput {
                field: "station_ladders",
                ..
            })
        ));
        assert!(matches!(
            model.evaluate_bem_convergence(
                &[refinement_stations(4), refinement_stations(8)],
                point,
                BladeElementBemOptions::default(),
                0.0,
            ),
            Err(BladeElementError::InvalidInput {
                field: "relative_load_tolerance",
                ..
            })
        ));
        let mut malformed = refinement_stations(8);
        malformed[3].radius_m = malformed[2].radius_m;
        let error = model
            .evaluate_bem_convergence(
                &[refinement_stations(4), malformed],
                point,
                BladeElementBemOptions::default(),
                0.05,
            )
            .expect_err("malformed refinement level must fail");
        assert!(
            matches!(
                error,
                BladeElementError::ConvergenceLevelFailed { level_index: 1, .. }
            ),
            "unexpected convergence error: {error:?}"
        );
    }

    #[test]
    fn zero_polar_coefficients_produce_zero_load() {
        let zero_polar = AirfoilPolar::try_new(vec![
            AirfoilPolarPoint {
                angle_of_attack_deg: -90.0,
                lift_coefficient: 0.0,
                drag_coefficient: 0.0,
            },
            AirfoilPolarPoint {
                angle_of_attack_deg: 90.0,
                lift_coefficient: 0.0,
                drag_coefficient: 0.0,
            },
        ])
        .expect("zero polar is valid");
        let zero_model = BladeElementModel::try_from_source(&source(), stations(24.0), zero_polar)
            .expect("zero model is valid");
        let result = zero_model
            .evaluate_at_freestream(10.0)
            .expect("zero-load point");
        assert_eq!(result.thrust_n, 0.0);
        assert_eq!(result.torque_nm, 0.0);
        assert_eq!(result.shaft_power_w, 0.0);
    }

    #[test]
    fn malformed_geometry_and_polar_fail_closed() {
        let mut invalid_stations = stations(24.0);
        invalid_stations[1].radius_m = invalid_stations[0].radius_m;
        assert!(matches!(
            BladeElementModel::try_from_source(&source(), invalid_stations, polar()),
            Err(BladeElementError::NonIncreasingStationRadius { .. })
        ));

        assert!(matches!(
            AirfoilPolar::try_new(vec![AirfoilPolarPoint {
                angle_of_attack_deg: 0.0,
                lift_coefficient: 0.0,
                drag_coefficient: -0.1,
            }]),
            Err(BladeElementError::InvalidInput { .. })
                | Err(BladeElementError::InvalidPolarPoint { .. })
        ));

        assert!(matches!(
            model().evaluate(BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: -1.0,
                density_kg_m3: 1.0,
            }),
            Err(BladeElementError::InvalidInput { .. })
        ));
        assert!(matches!(
            BladeElementModel::try_from_source(&source(), stations(80.0), polar(),)
                .expect("geometry is valid")
                .evaluate_at_freestream(0.0),
            Err(BladeElementError::AngleOfAttackOutsidePolar { .. })
        ));
    }
}
