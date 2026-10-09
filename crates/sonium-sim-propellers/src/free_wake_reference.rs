//! Checked fixed-circulation steady free-wake continuation.
//!
//! This module iteratively rolls up a discretized helical wake using
//! cross-filament Biot--Savart induction. Sectional circulation is supplied by
//! a converged BEM result, so this is a fixed-circulation steady free-wake
//! reference rather than a fully load-coupled free-wake solver. The explicit
//! convergence and operation budgets prevent a non-convergent or unbounded
//! wake from being promoted as a physical result. Viscous core growth,
//! unsteady shedding, aeroelasticity, installation, and CFD/FW-H remain out
//! of scope.

use std::f64::consts::{PI, TAU};
use std::fmt;

use crate::{BladeElementBemResult, BladeElementModel, Point3};

/// Controls for fixed-circulation steady free-wake roll-up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadyFreeWakeOptions {
    /// Number of wake revolutions retained behind the rotor.
    pub wake_turns: usize,
    /// Wake segments per revolution.
    pub segments_per_turn: usize,
    /// Regularization radius for finite vortex segments [m].
    pub core_radius_m: f64,
    /// Multiplier applied to local BEM axial convection speed.
    pub axial_convection_factor: f64,
    /// Fixed-point under-relaxation factor in `(0, 1]`.
    pub relaxation: f64,
    /// Maximum node displacement accepted at convergence [m].
    pub convergence_tolerance_m: f64,
    /// Maximum roll-up iterations.
    pub max_iterations: usize,
    /// Maximum total wake nodes.
    pub max_node_count: usize,
    /// Maximum Biot--Savart segment evaluations per iteration.
    pub max_biot_savart_evaluations: usize,
    /// Sign of the declared rotation direction (`+1` or `-1`).
    pub rotation_direction: f64,
}

impl Default for SteadyFreeWakeOptions {
    fn default() -> Self {
        Self {
            wake_turns: 2,
            segments_per_turn: 16,
            core_radius_m: 1.0e-3,
            axial_convection_factor: 1.0,
            relaxation: 0.25,
            convergence_tolerance_m: 1.0e-5,
            max_iterations: 40,
            max_node_count: 20_000,
            max_biot_savart_evaluations: 5_000_000,
            rotation_direction: 1.0,
        }
    }
}

/// One converged free-wake node and its last induced velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadyFreeWakeNode {
    /// Cartesian node position [m]. The rotor axis is `+z`.
    pub position: Point3,
    /// Cross-filament induced velocity used by the fixed-point update [m/s].
    pub induced_velocity_m_s: Point3,
    /// Wake age from shedding [s].
    pub age_s: f64,
}

/// One trailing-vortex filament in the converged wake.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyFreeWakeFilament {
    /// Blade index that shed this filament.
    pub blade_index: usize,
    /// Initial azimuth of the blade [rad].
    pub base_azimuth_rad: f64,
    /// Filament radius at shedding [m].
    pub radius_m: f64,
    /// Fixed circulation assigned from BEM spanwise loading [m²/s].
    pub circulation_m2_s: f64,
    /// Axial convection speed used by the baseline helix [m/s].
    pub axial_convection_speed_m_s: f64,
    /// Nodes in downstream age order, including the anchored shedding node.
    pub nodes: Vec<SteadyFreeWakeNode>,
}

/// Converged fixed-circulation steady free-wake result.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyFreeWakeResult {
    /// Operating point inherited from BEM.
    pub operating_point: crate::BladeElementOperatingPoint,
    /// Number of blades represented by the wake.
    pub blade_count: usize,
    /// Number of fixed-point updates accepted.
    pub iterations: usize,
    /// Final maximum node displacement [m].
    pub final_residual_m: f64,
    /// Whether the declared node displacement tolerance was met.
    pub converged: bool,
    /// Total nodes retained across all filaments.
    pub node_count: usize,
    /// Total Biot--Savart evaluations per iteration.
    pub biot_savart_evaluation_count: usize,
    /// Regularization radius used by the kernel [m].
    pub core_radius_m: f64,
    /// Converged blade-indexed trailing filaments.
    pub filaments: Vec<SteadyFreeWakeFilament>,
}

impl SteadyFreeWakeResult {
    /// Whether the converged result contains finite, internally consistent
    /// state and is safe to pass to a downstream diagnostic.
    pub fn is_finite(&self) -> bool {
        self.operating_point.rpm.is_finite()
            && self.operating_point.rpm > 0.0
            && self.operating_point.freestream_m_s.is_finite()
            && self.operating_point.freestream_m_s > 0.0
            && self.operating_point.density_kg_m3.is_finite()
            && self.operating_point.density_kg_m3 > 0.0
            && self.blade_count > 0
            && self.iterations > 0
            && self.final_residual_m.is_finite()
            && self.final_residual_m >= 0.0
            && self.converged
            && self.node_count > 0
            && self.biot_savart_evaluation_count > 0
            && self.core_radius_m.is_finite()
            && self.core_radius_m > 0.0
            && self.filaments.iter().all(|filament| {
                filament.blade_index < self.blade_count
                    && filament.base_azimuth_rad.is_finite()
                    && filament.radius_m.is_finite()
                    && filament.radius_m > 0.0
                    && filament.circulation_m2_s.is_finite()
                    && filament.axial_convection_speed_m_s.is_finite()
                    && filament.axial_convection_speed_m_s > 0.0
                    && !filament.nodes.is_empty()
                    && filament.nodes.iter().all(|node| {
                        finite_point(node.position)
                            && finite_point(node.induced_velocity_m_s)
                            && node.age_s.is_finite()
                            && node.age_s >= 0.0
                    })
            })
            && self.filaments.iter().map(|f| f.nodes.len()).sum::<usize>() == self.node_count
    }

    /// Evaluate the induced velocity at an arbitrary finite observer point.
    ///
    /// The returned field is a diagnostic of the converged fixed-circulation
    /// wake and is not an acoustic observer pressure.
    pub fn induced_velocity_at(&self, point: Point3) -> Result<Point3, SteadyFreeWakeError> {
        if !finite_point(point) {
            return Err(SteadyFreeWakeError::InvalidPoint {
                field: "observer_point",
            });
        }
        let mut induced = zero();
        for (filament_index, filament) in self.filaments.iter().enumerate() {
            for nodes in filament.nodes.windows(2) {
                let contribution = finite_vortex_segment_velocity(
                    point,
                    nodes[0].position,
                    nodes[1].position,
                    filament.circulation_m2_s,
                    self.core_radius_m,
                );
                if !finite_point(contribution) {
                    return Err(SteadyFreeWakeError::NonFiniteState {
                        filament: filament_index,
                        node: 0,
                    });
                }
                induced = add(induced, contribution);
            }
        }
        if !finite_point(induced) {
            return Err(SteadyFreeWakeError::InvalidPoint {
                field: "induced_velocity",
            });
        }
        Ok(induced)
    }
}

/// Failure-closed steady free-wake error.
#[derive(Debug, Clone, PartialEq)]
pub enum SteadyFreeWakeError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The BEM source does not satisfy the checked fixed-circulation contract.
    InvalidBemResult {
        /// Invalid source field or invariant.
        field: &'static str,
    },
    /// The model and result have different station counts.
    StationCountMismatch {
        /// Number of model stations.
        model_count: usize,
        /// Number of BEM load stations.
        result_count: usize,
    },
    /// The requested wake size or work budget overflows `usize`.
    SizeOverflow,
    /// The estimated kernel work exceeds the caller budget.
    EvaluationBudgetExceeded {
        /// Estimated evaluations per iteration.
        requested: usize,
        /// Caller budget.
        budget: usize,
    },
    /// A node or kernel state is non-finite.
    NonFiniteState {
        /// Filament index.
        filament: usize,
        /// Node index within that filament.
        node: usize,
    },
    /// An observer point or derived field is invalid.
    InvalidPoint {
        /// Field name.
        field: &'static str,
    },
    /// The fixed-point roll-up did not meet the requested displacement gate.
    NotConverged {
        /// Iterations performed.
        iterations: usize,
        /// Final maximum displacement [m].
        residual_m: f64,
    },
}

impl fmt::Display for SteadyFreeWakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(f, "steady free wake {field} must be {requirement}")
            }
            Self::InvalidBemResult { field } => {
                write!(f, "steady free wake BEM result {field} is invalid")
            }
            Self::StationCountMismatch {
                model_count,
                result_count,
            } => write!(
                f,
                "steady free wake station counts differ: model={model_count}, result={result_count}"
            ),
            Self::SizeOverflow => write!(f, "steady free wake size overflows usize"),
            Self::EvaluationBudgetExceeded { requested, budget } => write!(
                f,
                "steady free wake kernel budget exceeded: requested {requested}, budget {budget}"
            ),
            Self::NonFiniteState { filament, node } => write!(
                f,
                "steady free wake filament {filament} node {node} is non-finite"
            ),
            Self::InvalidPoint { field } => write!(f, "steady free wake {field} is invalid"),
            Self::NotConverged {
                iterations,
                residual_m,
            } => write!(
                f,
                "steady free wake did not converge in {iterations} iterations (residual {residual_m:e} m)"
            ),
        }
    }
}

impl std::error::Error for SteadyFreeWakeError {}

/// Roll up a fixed-circulation steady wake from a converged BEM result.
pub fn evaluate_steady_free_wake(
    model: &BladeElementModel,
    bem: &BladeElementBemResult,
    options: SteadyFreeWakeOptions,
) -> Result<SteadyFreeWakeResult, SteadyFreeWakeError> {
    validate_options(options)?;
    if !bem.converged || bem.max_iterations_used == 0 {
        return Err(SteadyFreeWakeError::InvalidBemResult { field: "converged" });
    }
    if bem.loads.strips.len() != bem.strips.len() {
        return Err(SteadyFreeWakeError::InvalidBemResult {
            field: "strip_diagnostics",
        });
    }
    let model_stations = model.stations();
    if model_stations.len() != bem.loads.strips.len() {
        return Err(SteadyFreeWakeError::StationCountMismatch {
            model_count: model_stations.len(),
            result_count: bem.loads.strips.len(),
        });
    }
    let operating_point = bem.loads.operating_point;
    if !operating_point.rpm.is_finite()
        || operating_point.rpm <= 0.0
        || !operating_point.freestream_m_s.is_finite()
        || operating_point.freestream_m_s <= 0.0
        || !operating_point.density_kg_m3.is_finite()
        || operating_point.density_kg_m3 <= 0.0
    {
        return Err(SteadyFreeWakeError::InvalidBemResult {
            field: "operating_point",
        });
    }
    let omega_rad_s = TAU * operating_point.rpm / 60.0;
    if !omega_rad_s.is_finite() || omega_rad_s <= 0.0 {
        return Err(SteadyFreeWakeError::InvalidBemResult {
            field: "omega_rad_s",
        });
    }

    let segment_count = options
        .wake_turns
        .checked_mul(options.segments_per_turn)
        .ok_or(SteadyFreeWakeError::SizeOverflow)?;
    let node_count_per_filament = segment_count
        .checked_add(1)
        .ok_or(SteadyFreeWakeError::SizeOverflow)?;
    let seed_count = model_stations
        .len()
        .checked_add(1)
        .ok_or(SteadyFreeWakeError::SizeOverflow)?;
    let filament_count = seed_count
        .checked_mul(model.n_blades())
        .ok_or(SteadyFreeWakeError::SizeOverflow)?;
    let node_count = filament_count
        .checked_mul(node_count_per_filament)
        .ok_or(SteadyFreeWakeError::SizeOverflow)?;
    if node_count > options.max_node_count {
        return Err(SteadyFreeWakeError::InvalidOptions {
            field: "node_count",
            requirement: "within max_node_count",
        });
    }
    let source_segment_count = filament_count
        .checked_mul(segment_count)
        .ok_or(SteadyFreeWakeError::SizeOverflow)?;
    let evaluation_count = node_count
        .checked_mul(source_segment_count)
        .ok_or(SteadyFreeWakeError::SizeOverflow)?;
    if evaluation_count > options.max_biot_savart_evaluations {
        return Err(SteadyFreeWakeError::EvaluationBudgetExceeded {
            requested: evaluation_count,
            budget: options.max_biot_savart_evaluations,
        });
    }

    let mut circulation = Vec::with_capacity(model_stations.len());
    let mut axial_speeds = Vec::with_capacity(model_stations.len());
    for (index, (station, load)) in model_stations.iter().zip(&bem.loads.strips).enumerate() {
        let induction = bem.strips[index];
        if !induction.axial_induction.is_finite()
            || !induction.tangential_induction.is_finite()
            || !induction.loss_factor.is_finite()
            || !induction.residual.is_finite()
            || induction.iterations == 0
        {
            return Err(SteadyFreeWakeError::InvalidBemResult {
                field: "induction_diagnostics",
            });
        }
        if !station.chord_m.is_finite()
            || station.chord_m <= 0.0
            || !load.relative_speed_m_s.is_finite()
            || load.relative_speed_m_s <= 0.0
            || !load.lift_coefficient.is_finite()
        {
            return Err(SteadyFreeWakeError::InvalidBemResult {
                field: "sectional_loading",
            });
        }
        let gamma = 0.5 * station.chord_m * load.relative_speed_m_s * load.lift_coefficient;
        let axial_speed = operating_point.freestream_m_s
            * (1.0 + induction.axial_induction)
            * options.axial_convection_factor;
        if !gamma.is_finite() || !axial_speed.is_finite() || axial_speed <= 0.0 {
            return Err(SteadyFreeWakeError::InvalidBemResult {
                field: "circulation_or_convection",
            });
        }
        circulation.push(gamma);
        axial_speeds.push(axial_speed);
    }

    let mut seeds = Vec::with_capacity(seed_count);
    let mut previous = 0.0;
    for (index, station) in model_stations.iter().enumerate() {
        seeds.push(WakeSeed {
            radius_m: station.radius_m,
            circulation_m2_s: circulation[index] - previous,
            axial_speed_m_s: axial_speeds[index],
        });
        previous = circulation[index];
    }
    seeds.push(WakeSeed {
        radius_m: model.tip_radius_m(),
        circulation_m2_s: -previous,
        axial_speed_m_s: *axial_speeds
            .last()
            .ok_or(SteadyFreeWakeError::InvalidBemResult {
                field: "axial_induction",
            })?,
    });

    let mut filaments = Vec::with_capacity(filament_count);
    for blade_index in 0..model.n_blades() {
        let base_azimuth_rad = TAU * blade_index as f64 / model.n_blades() as f64;
        for seed in &seeds {
            let nodes = (0..=segment_count)
                .map(|node_index| {
                    let age_s =
                        TAU * node_index as f64 / (options.segments_per_turn as f64 * omega_rad_s);
                    SteadyFreeWakeNode {
                        position: helical_position(
                            seed.radius_m,
                            base_azimuth_rad,
                            age_s,
                            seed.axial_speed_m_s,
                            omega_rad_s,
                            options.rotation_direction,
                        ),
                        induced_velocity_m_s: zero(),
                        age_s,
                    }
                })
                .collect::<Vec<_>>();
            filaments.push(SteadyFreeWakeFilament {
                blade_index,
                base_azimuth_rad,
                radius_m: seed.radius_m,
                circulation_m2_s: seed.circulation_m2_s,
                axial_convection_speed_m_s: seed.axial_speed_m_s,
                nodes,
            });
        }
    }

    let mut final_residual_m = f64::INFINITY;
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 1..=options.max_iterations {
        let mut next = filaments.clone();
        let mut residual_m: f64 = 0.0;
        for (filament_index, filament) in filaments.iter().enumerate() {
            for (node_index, node) in filament.nodes.iter().enumerate() {
                if node_index == 0 {
                    next[filament_index].nodes[node_index].induced_velocity_m_s = zero();
                    continue;
                }
                let induced = induced_from_other_filaments(
                    filament_index,
                    node.position,
                    &filaments,
                    options.core_radius_m,
                );
                if !finite_point(induced) {
                    return Err(SteadyFreeWakeError::NonFiniteState {
                        filament: filament_index,
                        node: node_index,
                    });
                }
                let baseline = helical_position(
                    filament.radius_m,
                    filament.base_azimuth_rad,
                    node.age_s,
                    filament.axial_convection_speed_m_s,
                    omega_rad_s,
                    options.rotation_direction,
                );
                let target = add(baseline, scale(induced, node.age_s));
                let updated = add(
                    node.position,
                    scale(sub(target, node.position), options.relaxation),
                );
                let displacement = norm(sub(updated, node.position));
                if !displacement.is_finite() {
                    return Err(SteadyFreeWakeError::NonFiniteState {
                        filament: filament_index,
                        node: node_index,
                    });
                }
                residual_m = residual_m.max(displacement);
                next[filament_index].nodes[node_index] = SteadyFreeWakeNode {
                    position: updated,
                    induced_velocity_m_s: induced,
                    age_s: node.age_s,
                };
            }
        }
        filaments = next;
        iterations = iteration;
        final_residual_m = residual_m;
        if residual_m <= options.convergence_tolerance_m {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SteadyFreeWakeError::NotConverged {
            iterations,
            residual_m: final_residual_m,
        });
    }

    let result = SteadyFreeWakeResult {
        operating_point,
        blade_count: model.n_blades(),
        iterations,
        final_residual_m,
        converged,
        node_count,
        biot_savart_evaluation_count: evaluation_count,
        core_radius_m: options.core_radius_m,
        filaments,
    };
    if !result.is_finite() {
        return Err(SteadyFreeWakeError::NonFiniteState {
            filament: 0,
            node: 0,
        });
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy)]
struct WakeSeed {
    radius_m: f64,
    circulation_m2_s: f64,
    axial_speed_m_s: f64,
}

fn validate_options(options: SteadyFreeWakeOptions) -> Result<(), SteadyFreeWakeError> {
    if options.wake_turns == 0 {
        return Err(SteadyFreeWakeError::InvalidOptions {
            field: "wake_turns",
            requirement: "non-zero",
        });
    }
    if options.segments_per_turn < 2 {
        return Err(SteadyFreeWakeError::InvalidOptions {
            field: "segments_per_turn",
            requirement: "at least two",
        });
    }
    if options.max_iterations == 0 {
        return Err(SteadyFreeWakeError::InvalidOptions {
            field: "max_iterations",
            requirement: "non-zero",
        });
    }
    for (field, value) in [
        ("core_radius_m", options.core_radius_m),
        ("axial_convection_factor", options.axial_convection_factor),
        ("relaxation", options.relaxation),
        ("convergence_tolerance_m", options.convergence_tolerance_m),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(SteadyFreeWakeError::InvalidOptions {
                field,
                requirement: "finite and positive",
            });
        }
    }
    if options.relaxation > 1.0 {
        return Err(SteadyFreeWakeError::InvalidOptions {
            field: "relaxation",
            requirement: "within (0, 1]",
        });
    }
    if options.max_node_count == 0 || options.max_biot_savart_evaluations == 0 {
        return Err(SteadyFreeWakeError::InvalidOptions {
            field: "max_node_count/max_biot_savart_evaluations",
            requirement: "non-zero",
        });
    }
    if options.rotation_direction != 1.0 && options.rotation_direction != -1.0 {
        return Err(SteadyFreeWakeError::InvalidOptions {
            field: "rotation_direction",
            requirement: "exactly +1 or -1",
        });
    }
    Ok(())
}

fn helical_position(
    radius_m: f64,
    base_azimuth_rad: f64,
    age_s: f64,
    axial_speed_m_s: f64,
    omega_rad_s: f64,
    rotation_direction: f64,
) -> Point3 {
    let wake_angle = omega_rad_s * age_s;
    let azimuth = base_azimuth_rad + rotation_direction * wake_angle;
    Point3 {
        x: radius_m * azimuth.cos(),
        y: radius_m * azimuth.sin(),
        z: axial_speed_m_s * age_s,
    }
}

fn induced_from_other_filaments(
    target_filament: usize,
    point: Point3,
    filaments: &[SteadyFreeWakeFilament],
    core_radius_m: f64,
) -> Point3 {
    let mut induced = zero();
    for (filament_index, filament) in filaments.iter().enumerate() {
        if filament_index == target_filament {
            continue;
        }
        for nodes in filament.nodes.windows(2) {
            induced = add(
                induced,
                finite_vortex_segment_velocity(
                    point,
                    nodes[0].position,
                    nodes[1].position,
                    filament.circulation_m2_s,
                    core_radius_m,
                ),
            );
        }
    }
    induced
}

fn finite_vortex_segment_velocity(
    receiver: Point3,
    start: Point3,
    end: Point3,
    circulation_m2_s: f64,
    core_radius_m: f64,
) -> Point3 {
    let r1 = sub(receiver, start);
    let r2 = sub(receiver, end);
    let segment = sub(end, start);
    let r1_norm = norm(r1).max(core_radius_m);
    let r2_norm = norm(r2).max(core_radius_m);
    let cross_product = cross(r1, r2);
    let denominator =
        dot(cross_product, cross_product) + core_radius_m * core_radius_m * dot(segment, segment);
    if denominator <= f64::MIN_POSITIVE || !denominator.is_finite() {
        return zero();
    }
    let direction_difference = sub(scale(r1, 1.0 / r1_norm), scale(r2, 1.0 / r2_norm));
    let coefficient =
        circulation_m2_s / (4.0 * PI) * dot(segment, direction_difference) / denominator;
    scale(cross_product, coefficient)
}

fn zero() -> Point3 {
    Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    }
}

fn add(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x + right.x,
        y: left.y + right.y,
        z: left.z + right.z,
    }
}

fn sub(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn scale(value: Point3, factor: f64) -> Point3 {
    Point3 {
        x: value.x * factor,
        y: value.y * factor,
        z: value.z * factor,
    }
}

fn dot(left: Point3, right: Point3) -> f64 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn cross(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.y * right.z - left.z * right.y,
        y: left.z * right.x - left.x * right.z,
        z: left.x * right.y - left.y * right.x,
    }
}

fn norm(value: Point3) -> f64 {
    value.x.hypot(value.y).hypot(value.z)
}

fn finite_point(value: Point3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementStation};
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
        .expect("valid free-wake polar");
        BladeElementModel::try_from_source(&source, stations, polar).expect("valid free-wake model")
    }

    fn bem(model: &BladeElementModel) -> BladeElementBemResult {
        model
            .evaluate_bem(
                crate::BladeElementOperatingPoint {
                    rpm: 12_000.0,
                    freestream_m_s: 10.0,
                    density_kg_m3: 1.204,
                },
                BladeElementBemOptions::default(),
            )
            .expect("converged free-wake BEM")
    }

    fn options() -> SteadyFreeWakeOptions {
        SteadyFreeWakeOptions {
            wake_turns: 1,
            segments_per_turn: 8,
            max_iterations: 100,
            relaxation: 0.05,
            convergence_tolerance_m: 1.0e-4,
            max_node_count: 10_000,
            max_biot_savart_evaluations: 1_000_000,
            ..SteadyFreeWakeOptions::default()
        }
    }

    #[test]
    fn free_wake_rollup_is_deterministic_converged_and_finite() {
        let model = model();
        let first =
            evaluate_steady_free_wake(&model, &bem(&model), options()).expect("free-wake roll-up");
        let second = evaluate_steady_free_wake(&model, &bem(&model), options())
            .expect("deterministic free-wake roll-up");
        assert_eq!(first, second);
        assert!(first.converged);
        assert!(first.is_finite());
        assert!(first.final_residual_m <= options().convergence_tolerance_m);
        assert_eq!(
            first.node_count,
            first.filaments.iter().map(|f| f.nodes.len()).sum::<usize>()
        );
        assert!(first.filaments.iter().any(|filament| {
            filament
                .nodes
                .iter()
                .skip(1)
                .any(|node| node.position.x != filament.radius_m || node.position.y != 0.0)
        }));
    }

    #[test]
    fn zero_circulation_free_wake_matches_baseline_helix_and_has_zero_induction() {
        let model = model();
        let mut zero_bem = bem(&model);
        for strip in &mut zero_bem.loads.strips {
            strip.lift_coefficient = 0.0;
        }
        let result = evaluate_steady_free_wake(&model, &zero_bem, options())
            .expect("zero-circulation free wake");
        assert_eq!(result.final_residual_m, 0.0);
        assert!(result.filaments.iter().all(|filament| {
            filament
                .nodes
                .iter()
                .all(|node| node.induced_velocity_m_s == zero())
        }));
        let field = result
            .induced_velocity_at(Point3 {
                x: 0.12,
                y: 0.0,
                z: 0.02,
            })
            .expect("finite zero field");
        assert_eq!(field, zero());
    }

    #[test]
    fn free_wake_observer_field_and_circulation_scaling_are_explicit() {
        let model = model();
        let base_bem = bem(&model);
        let base = evaluate_steady_free_wake(&model, &base_bem, options()).expect("base wake");
        let observer = Point3 {
            x: 0.12,
            y: 0.03,
            z: 0.05,
        };
        let base_field = base.induced_velocity_at(observer).expect("base field");
        assert!(finite_point(base_field));
        let mut doubled_bem = base_bem;
        for strip in &mut doubled_bem.loads.strips {
            strip.lift_coefficient *= 2.0;
        }
        let doubled_options = SteadyFreeWakeOptions {
            relaxation: 0.02,
            max_iterations: 300,
            ..options()
        };
        let doubled = evaluate_steady_free_wake(&model, &doubled_bem, doubled_options)
            .expect("doubled circulation wake");
        let doubled_field = doubled
            .induced_velocity_at(observer)
            .expect("doubled field");
        let scale = norm(base_field).max(1.0e-12);
        assert!((norm(doubled_field) / scale - 2.0).abs() < 5.0e-2);
        for (base_filament, doubled_filament) in base.filaments.iter().zip(&doubled.filaments) {
            assert!(
                (doubled_filament.circulation_m2_s - 2.0 * base_filament.circulation_m2_s).abs()
                    < 1.0e-12
            );
        }
    }

    #[test]
    fn free_wake_rejects_invalid_options_inputs_and_budgets() {
        let model = model();
        let bem = bem(&model);
        for invalid in [
            SteadyFreeWakeOptions {
                wake_turns: 0,
                ..options()
            },
            SteadyFreeWakeOptions {
                segments_per_turn: 1,
                ..options()
            },
            SteadyFreeWakeOptions {
                relaxation: 1.1,
                ..options()
            },
            SteadyFreeWakeOptions {
                max_iterations: 0,
                ..options()
            },
            SteadyFreeWakeOptions {
                max_node_count: 1,
                ..options()
            },
        ] {
            assert!(matches!(
                evaluate_steady_free_wake(&model, &bem, invalid),
                Err(SteadyFreeWakeError::InvalidOptions { .. })
            ));
        }
        assert!(matches!(
            evaluate_steady_free_wake(
                &model,
                &bem,
                SteadyFreeWakeOptions {
                    max_biot_savart_evaluations: 1,
                    ..options()
                },
            ),
            Err(SteadyFreeWakeError::EvaluationBudgetExceeded { .. })
        ));
        let mut unconverged = bem.clone();
        unconverged.converged = false;
        assert!(matches!(
            evaluate_steady_free_wake(&model, &unconverged, options()),
            Err(SteadyFreeWakeError::InvalidBemResult { field: "converged" })
        ));
        assert!(matches!(
            evaluate_steady_free_wake(
                &model,
                &bem,
                SteadyFreeWakeOptions {
                    max_iterations: 1,
                    convergence_tolerance_m: 1.0e-20,
                    ..options()
                },
            ),
            Err(SteadyFreeWakeError::NotConverged { .. })
        ));
    }

    #[test]
    fn free_wake_kernel_rejects_nonfinite_observer_points() {
        let model = model();
        let result =
            evaluate_steady_free_wake(&model, &bem(&model), options()).expect("free-wake result");
        assert!(matches!(
            result.induced_velocity_at(Point3 {
                x: f64::NAN,
                y: 0.0,
                z: 0.0,
            }),
            Err(SteadyFreeWakeError::InvalidPoint {
                field: "observer_point"
            })
        ));
    }
}
