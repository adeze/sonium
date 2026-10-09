//! Checked prescribed-helical-wake reference for propeller studies.
//!
//! This module turns converged local BEM circulation into a spatially
//! discretized helical trailing-vortex reference and evaluates its induced
//! velocity with a regularized finite-segment Biot--Savart kernel. It is a
//! useful intermediate fidelity level between local BEM and a free-wake
//! solver: the wake is prescribed from the BEM state and is not deformed by a
//! coupled wake iteration. Unsteady shedding, viscous core growth, skew
//! corrections, blade elasticity, installation, and CFD/FW-H remain outside
//! this contract.

use std::f64::consts::{PI, TAU};
use std::fmt;

use crate::{BladeElementBemResult, BladeElementModel, Point3};

/// Controls for the prescribed helical wake discretization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrescribedHelicalWakeOptions {
    /// Number of revolutions retained behind the rotor.
    pub wake_turns: usize,
    /// Finite vortex segments per retained revolution.
    pub segments_per_turn: usize,
    /// Positive viscous-core regularization radius [m].
    pub core_radius_m: f64,
    /// Multiplier applied to local BEM axial convection speed.
    pub axial_convection_factor: f64,
    /// Downstream distance of the receiver plane from the rotor [m].
    pub evaluation_offset_m: f64,
    /// Sign of the declared rotation direction (`+1` or `-1`).
    pub rotation_direction: f64,
    /// Maximum number of finite vortex segments retained in one evaluation.
    pub max_segment_count: usize,
    /// Maximum number of receiver/filament/segment kernel evaluations.
    pub max_biot_savart_evaluations: usize,
}

impl Default for PrescribedHelicalWakeOptions {
    fn default() -> Self {
        Self {
            wake_turns: 4,
            segments_per_turn: 32,
            core_radius_m: 1.0e-3,
            axial_convection_factor: 1.0,
            evaluation_offset_m: 2.0e-3,
            rotation_direction: 1.0,
            max_segment_count: 1_000_000,
            max_biot_savart_evaluations: 5_000_000,
        }
    }
}

/// Induced velocity and circulation evidence at one BEM radial station.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrescribedHelicalWakeStation {
    /// Station radius [m].
    pub radius_m: f64,
    /// Bound circulation inferred from the BEM sectional lift [m²/s].
    pub circulation_m2_s: f64,
    /// Discrete radial circulation change shed into the wake [m²/s].
    pub trailing_circulation_m2_s: f64,
    /// Local helical wake pitch for one revolution [m/rev].
    pub wake_pitch_m_per_rev: f64,
    /// Induced velocity in the rotor axial direction [m/s].
    pub induced_axial_velocity_m_s: f64,
    /// Induced velocity in the positive tangential direction at the receiver
    /// azimuth [m/s].
    pub induced_tangential_velocity_m_s: f64,
    /// Induced velocity in the outward radial direction at the receiver
    /// azimuth [m/s].
    pub induced_radial_velocity_m_s: f64,
    /// Magnitude of the three-component induced velocity [m/s].
    pub induced_speed_m_s: f64,
}

/// Result of the prescribed helical-wake reference.
#[derive(Debug, Clone, PartialEq)]
pub struct PrescribedHelicalWakeResult {
    /// Operating point inherited from the converged BEM result.
    pub operating_point: crate::BladeElementOperatingPoint,
    /// Number of blades represented by the discretized wake.
    pub blade_count: usize,
    /// Receiver-plane offset used for all station velocities [m].
    pub evaluation_offset_m: f64,
    /// Number of finite vortex segments evaluated per receiver station.
    pub segment_count: usize,
    /// Total finite-segment kernel evaluations performed.
    pub biot_savart_evaluation_count: usize,
    /// Station-level circulation and induced-velocity evidence.
    pub stations: Vec<PrescribedHelicalWakeStation>,
    /// Largest induced speed over all receiver stations [m/s].
    pub maximum_induced_speed_m_s: f64,
}

impl PrescribedHelicalWakeResult {
    /// Whether all retained wake evidence is finite and physically bounded.
    pub fn is_finite(&self) -> bool {
        self.operating_point.rpm.is_finite()
            && self.operating_point.rpm > 0.0
            && self.operating_point.freestream_m_s.is_finite()
            && self.operating_point.freestream_m_s > 0.0
            && self.operating_point.density_kg_m3.is_finite()
            && self.operating_point.density_kg_m3 > 0.0
            && self.blade_count > 0
            && self.evaluation_offset_m.is_finite()
            && self.evaluation_offset_m > 0.0
            && self.segment_count > 0
            && self.biot_savart_evaluation_count > 0
            && self.maximum_induced_speed_m_s.is_finite()
            && self.maximum_induced_speed_m_s >= 0.0
            && self.stations.iter().all(|station| {
                station.radius_m.is_finite()
                    && station.radius_m > 0.0
                    && station.circulation_m2_s.is_finite()
                    && station.trailing_circulation_m2_s.is_finite()
                    && station.wake_pitch_m_per_rev.is_finite()
                    && station.wake_pitch_m_per_rev > 0.0
                    && station.induced_axial_velocity_m_s.is_finite()
                    && station.induced_tangential_velocity_m_s.is_finite()
                    && station.induced_radial_velocity_m_s.is_finite()
                    && station.induced_speed_m_s.is_finite()
                    && station.induced_speed_m_s >= 0.0
            })
    }
}

/// Failure-closed input or numerical error for the prescribed wake.
#[derive(Debug, Clone, PartialEq)]
pub enum PrescribedHelicalWakeError {
    /// A wake option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A BEM result is not suitable as a checked wake source.
    InvalidBemResult {
        /// Result field or invariant that failed.
        field: &'static str,
    },
    /// The model and BEM result have different station layouts.
    StationCountMismatch {
        /// Number of model stations.
        model_count: usize,
        /// Number of BEM load stations.
        result_count: usize,
    },
    /// A station input or derived circulation is not finite/physical.
    InvalidStation {
        /// Station index.
        index: usize,
        /// Field that failed admission.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The requested finite-segment count overflowed `usize`.
    SegmentCountOverflow,
    /// A Biot--Savart segment produced a non-finite contribution.
    NonFiniteContribution {
        /// Receiver station index.
        station: usize,
    },
}

impl fmt::Display for PrescribedHelicalWakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(f, "prescribed helical wake {field} must be {requirement}")
            }
            Self::InvalidBemResult { field } => {
                write!(f, "prescribed helical wake BEM result {field} is invalid")
            }
            Self::StationCountMismatch {
                model_count,
                result_count,
            } => write!(
                f,
                "prescribed helical wake station counts differ: model={model_count}, result={result_count}"
            ),
            Self::InvalidStation {
                index,
                field,
                requirement,
            } => write!(
                f,
                "prescribed helical wake station {index} {field} must be {requirement}"
            ),
            Self::SegmentCountOverflow => {
                write!(f, "prescribed helical wake segment count overflows usize")
            }
            Self::NonFiniteContribution { station } => write!(
                f,
                "prescribed helical wake station {station} has a non-finite induced-velocity contribution"
            ),
        }
    }
}

impl std::error::Error for PrescribedHelicalWakeError {}

/// Evaluate a regularized prescribed helical wake from converged BEM loads.
///
/// The wake uses one trailing filament for each discrete circulation change,
/// one copy per blade, and a tip-closure filament. The receiver plane is at a
/// positive axial offset and uses the first-blade azimuth as its radial/
/// tangential frame. This is deterministic and spatially resolved, but it is
/// deliberately not a coupled free-wake or unsteady solution.
pub fn evaluate_prescribed_helical_wake(
    model: &BladeElementModel,
    bem: &BladeElementBemResult,
    options: PrescribedHelicalWakeOptions,
) -> Result<PrescribedHelicalWakeResult, PrescribedHelicalWakeError> {
    validate_options(options)?;
    if !bem.converged || bem.max_iterations_used == 0 {
        return Err(PrescribedHelicalWakeError::InvalidBemResult { field: "converged" });
    }
    if bem.loads.strips.len() != bem.strips.len() {
        return Err(PrescribedHelicalWakeError::InvalidBemResult {
            field: "strip_diagnostics",
        });
    }
    let model_stations = model.stations();
    if model_stations.len() != bem.loads.strips.len() {
        return Err(PrescribedHelicalWakeError::StationCountMismatch {
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
        return Err(PrescribedHelicalWakeError::InvalidBemResult {
            field: "operating_point",
        });
    }

    let omega_rad_s = TAU * operating_point.rpm / 60.0;
    if !omega_rad_s.is_finite() || omega_rad_s <= 0.0 {
        return Err(PrescribedHelicalWakeError::InvalidBemResult {
            field: "omega_rad_s",
        });
    }
    let segment_count = options
        .wake_turns
        .checked_mul(options.segments_per_turn)
        .ok_or(PrescribedHelicalWakeError::SegmentCountOverflow)?;
    if segment_count > options.max_segment_count {
        return Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "segment_count",
            requirement: "within max_segment_count",
        });
    }
    let biot_savart_evaluation_count = segment_count
        .checked_mul(model_stations.len().saturating_add(1))
        .and_then(|count| count.checked_mul(model.n_blades()))
        .and_then(|count| count.checked_mul(model_stations.len()))
        .ok_or(PrescribedHelicalWakeError::SegmentCountOverflow)?;
    if biot_savart_evaluation_count > options.max_biot_savart_evaluations {
        return Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "biot_savart_evaluations",
            requirement: "within max_biot_savart_evaluations",
        });
    }
    for strip in &bem.strips {
        if !strip.axial_induction.is_finite()
            || !strip.tangential_induction.is_finite()
            || !strip.loss_factor.is_finite()
            || !strip.residual.is_finite()
            || strip.iterations == 0
        {
            return Err(PrescribedHelicalWakeError::InvalidBemResult {
                field: "induction_diagnostics",
            });
        }
    }
    let axial_speeds = bem
        .strips
        .iter()
        .map(|strip| {
            operating_point.freestream_m_s
                * (1.0 + strip.axial_induction)
                * options.axial_convection_factor
        })
        .collect::<Vec<_>>();
    if axial_speeds
        .iter()
        .any(|speed| !speed.is_finite() || *speed <= 0.0)
    {
        return Err(PrescribedHelicalWakeError::InvalidBemResult {
            field: "axial_induction",
        });
    }

    let mut circulation = Vec::with_capacity(model_stations.len());
    for (index, (station, load)) in model_stations.iter().zip(&bem.loads.strips).enumerate() {
        for (field, value, requirement) in [
            ("radius_m", station.radius_m, "finite and positive"),
            ("chord_m", station.chord_m, "finite and positive"),
            (
                "relative_speed_m_s",
                load.relative_speed_m_s,
                "finite and positive",
            ),
            ("lift_coefficient", load.lift_coefficient, "finite"),
        ] {
            if !value.is_finite() || (field != "lift_coefficient" && value <= 0.0) {
                return Err(PrescribedHelicalWakeError::InvalidStation {
                    index,
                    field,
                    requirement,
                });
            }
        }
        let gamma = 0.5 * load.relative_speed_m_s * station.chord_m * load.lift_coefficient;
        if !gamma.is_finite() {
            return Err(PrescribedHelicalWakeError::InvalidStation {
                index,
                field: "circulation_m2_s",
                requirement: "finite",
            });
        }
        circulation.push(gamma);
    }

    let mut filaments = Vec::with_capacity(circulation.len() + 1);
    let mut previous_gamma = 0.0;
    for (index, (station, &gamma)) in model_stations.iter().zip(&circulation).enumerate() {
        filaments.push(WakeFilament {
            radius_m: station.radius_m,
            circulation_m2_s: gamma - previous_gamma,
            axial_speed_m_s: axial_speeds[index],
        });
        previous_gamma = gamma;
    }
    filaments.push(WakeFilament {
        radius_m: model.tip_radius_m(),
        circulation_m2_s: -previous_gamma,
        axial_speed_m_s: *axial_speeds.last().ok_or(
            PrescribedHelicalWakeError::InvalidBemResult {
                field: "axial_induction",
            },
        )?,
    });

    let mut stations = Vec::with_capacity(model_stations.len());
    let mut maximum_induced_speed_m_s: f64 = 0.0;
    for (index, ((station, &gamma), &axial_speed)) in model_stations
        .iter()
        .zip(&circulation)
        .zip(&axial_speeds)
        .enumerate()
    {
        let receiver = Point3 {
            x: station.radius_m,
            y: 0.0,
            z: options.evaluation_offset_m,
        };
        let mut induced = Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        for blade_index in 0..model.n_blades() {
            let base_azimuth = TAU * blade_index as f64 / model.n_blades() as f64;
            for filament in &filaments {
                let strength = filament.circulation_m2_s;
                if strength.abs() <= f64::MIN_POSITIVE {
                    continue;
                }
                for segment_index in 0..segment_count {
                    let start = wake_point(
                        filament.radius_m,
                        base_azimuth,
                        filament.axial_speed_m_s,
                        omega_rad_s,
                        options.rotation_direction,
                        options.segments_per_turn,
                        segment_index,
                    );
                    let end = wake_point(
                        filament.radius_m,
                        base_azimuth,
                        filament.axial_speed_m_s,
                        omega_rad_s,
                        options.rotation_direction,
                        options.segments_per_turn,
                        segment_index + 1,
                    );
                    let contribution = finite_vortex_segment_velocity(
                        receiver,
                        start,
                        end,
                        strength,
                        options.core_radius_m,
                    );
                    if !is_finite_point(contribution) {
                        return Err(PrescribedHelicalWakeError::NonFiniteContribution {
                            station: index,
                        });
                    }
                    induced = add(induced, contribution);
                }
            }
        }
        let induced_speed_m_s = norm(induced);
        if !induced_speed_m_s.is_finite() {
            return Err(PrescribedHelicalWakeError::NonFiniteContribution { station: index });
        }
        maximum_induced_speed_m_s = maximum_induced_speed_m_s.max(induced_speed_m_s);
        let trailing_circulation_m2_s = filaments[index].circulation_m2_s;
        let wake_pitch_m_per_rev = TAU * axial_speed / omega_rad_s;
        stations.push(PrescribedHelicalWakeStation {
            radius_m: station.radius_m,
            circulation_m2_s: gamma,
            trailing_circulation_m2_s,
            wake_pitch_m_per_rev,
            induced_axial_velocity_m_s: induced.z,
            induced_tangential_velocity_m_s: induced.y,
            induced_radial_velocity_m_s: induced.x,
            induced_speed_m_s,
        });
    }

    let result = PrescribedHelicalWakeResult {
        operating_point,
        blade_count: model.n_blades(),
        evaluation_offset_m: options.evaluation_offset_m,
        segment_count,
        biot_savart_evaluation_count,
        stations,
        maximum_induced_speed_m_s,
    };
    if !result.is_finite() {
        return Err(PrescribedHelicalWakeError::InvalidBemResult {
            field: "wake_result",
        });
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy)]
struct WakeFilament {
    radius_m: f64,
    circulation_m2_s: f64,
    axial_speed_m_s: f64,
}

fn validate_options(
    options: PrescribedHelicalWakeOptions,
) -> Result<(), PrescribedHelicalWakeError> {
    if options.wake_turns == 0 {
        return Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "wake_turns",
            requirement: "non-zero",
        });
    }
    if options.segments_per_turn < 2 {
        return Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "segments_per_turn",
            requirement: "at least two",
        });
    }
    for (field, value) in [
        ("core_radius_m", options.core_radius_m),
        ("axial_convection_factor", options.axial_convection_factor),
        ("evaluation_offset_m", options.evaluation_offset_m),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PrescribedHelicalWakeError::InvalidOptions {
                field,
                requirement: "finite and positive",
            });
        }
    }
    if options.max_segment_count == 0 {
        return Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "max_segment_count",
            requirement: "non-zero",
        });
    }
    if options.max_biot_savart_evaluations == 0 {
        return Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "max_biot_savart_evaluations",
            requirement: "non-zero",
        });
    }
    if options.rotation_direction != 1.0 && options.rotation_direction != -1.0 {
        return Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "rotation_direction",
            requirement: "exactly +1 or -1",
        });
    }
    Ok(())
}

fn wake_point(
    radius_m: f64,
    base_azimuth: f64,
    axial_speed_m_s: f64,
    omega_rad_s: f64,
    rotation_direction: f64,
    segments_per_turn: usize,
    segment_index: usize,
) -> Point3 {
    let wake_angle = TAU * segment_index as f64 / segments_per_turn as f64;
    let azimuth = base_azimuth + rotation_direction * wake_angle;
    let axial = axial_speed_m_s / omega_rad_s * wake_angle;
    Point3 {
        x: radius_m * azimuth.cos(),
        y: radius_m * azimuth.sin(),
        z: axial,
    }
}

/// Regularized finite-segment Biot--Savart velocity.
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
    let cross = cross(r1, r2);
    let denominator = dot(cross, cross) + core_radius_m * core_radius_m * dot(segment, segment);
    if denominator <= f64::MIN_POSITIVE || !denominator.is_finite() {
        return Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }
    let direction_difference = sub(scale(r1, 1.0 / r1_norm), scale(r2, 1.0 / r2_norm));
    let coefficient =
        circulation_m2_s / (4.0 * PI) * dot(segment, direction_difference) / denominator;
    scale(cross, coefficient)
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

fn is_finite_point(value: Point3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementStation};
    use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

    fn source() -> PropellerSource {
        PropellerSource::drone_5inch(12_000.0)
    }

    fn model() -> BladeElementModel {
        let stations = [
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
        .expect("valid test polar");
        BladeElementModel::try_from_source(&source(), stations.to_vec(), polar)
            .expect("valid test model")
    }

    fn bem() -> BladeElementBemResult {
        model()
            .evaluate_bem(
                crate::BladeElementOperatingPoint {
                    rpm: 12_000.0,
                    freestream_m_s: 10.0,
                    density_kg_m3: 1.204,
                },
                BladeElementBemOptions::default(),
            )
            .expect("converged test BEM")
    }

    #[test]
    fn prescribed_wake_is_deterministic_finite_and_point_associated() {
        let model = model();
        let result = evaluate_prescribed_helical_wake(
            &model,
            &bem(),
            PrescribedHelicalWakeOptions::default(),
        )
        .expect("prescribed wake");
        let replay = evaluate_prescribed_helical_wake(
            &model,
            &bem(),
            PrescribedHelicalWakeOptions::default(),
        )
        .expect("deterministic replay");
        assert_eq!(result, replay);
        assert!(result.is_finite());
        assert_eq!(result.stations.len(), model.stations().len());
        assert_eq!(result.blade_count, model.n_blades());
        assert!(result.segment_count > 0);
        assert!(result.biot_savart_evaluation_count > result.segment_count);
        assert!(result.maximum_induced_speed_m_s > 0.0);
        assert!(result.stations.iter().all(|station| {
            station.wake_pitch_m_per_rev > 0.0 && station.induced_speed_m_s.is_finite()
        }));
    }

    #[test]
    fn prescribed_wake_circulation_and_velocity_scale_linearly() {
        let model = model();
        let base_bem = bem();
        let base = evaluate_prescribed_helical_wake(
            &model,
            &base_bem,
            PrescribedHelicalWakeOptions {
                wake_turns: 2,
                segments_per_turn: 24,
                ..PrescribedHelicalWakeOptions::default()
            },
        )
        .expect("base wake");
        let mut doubled_bem = base_bem.clone();
        for strip in &mut doubled_bem.loads.strips {
            strip.lift_coefficient *= 2.0;
        }
        let doubled = evaluate_prescribed_helical_wake(
            &model,
            &doubled_bem,
            PrescribedHelicalWakeOptions {
                wake_turns: 2,
                segments_per_turn: 24,
                ..PrescribedHelicalWakeOptions::default()
            },
        )
        .expect("scaled wake");
        for (base_station, doubled_station) in base.stations.iter().zip(&doubled.stations) {
            assert!(
                (doubled_station.circulation_m2_s - 2.0 * base_station.circulation_m2_s).abs()
                    < 1.0e-12
            );
            let scale = base_station.induced_speed_m_s.max(1.0e-12);
            assert!((doubled_station.induced_speed_m_s / scale - 2.0).abs() < 1.0e-9);
        }
    }

    #[test]
    fn prescribed_wake_zero_circulation_has_zero_induced_velocity() {
        let model = model();
        let mut zero_bem = bem();
        for strip in &mut zero_bem.loads.strips {
            strip.lift_coefficient = 0.0;
        }
        let result = evaluate_prescribed_helical_wake(
            &model,
            &zero_bem,
            PrescribedHelicalWakeOptions::default(),
        )
        .expect("zero-circulation wake");
        assert_eq!(result.maximum_induced_speed_m_s, 0.0);
        assert!(
            result
                .stations
                .iter()
                .all(|station| station.induced_speed_m_s == 0.0)
        );
    }

    #[test]
    fn prescribed_wake_discretization_and_direction_are_explicit() {
        let model = model();
        let result = evaluate_prescribed_helical_wake(
            &model,
            &bem(),
            PrescribedHelicalWakeOptions {
                wake_turns: 1,
                segments_per_turn: 12,
                rotation_direction: -1.0,
                ..PrescribedHelicalWakeOptions::default()
            },
        )
        .expect("reverse-direction wake");
        assert_eq!(result.segment_count, 12);
        assert!(result.stations.iter().all(|station| {
            station.wake_pitch_m_per_rev.is_finite() && station.wake_pitch_m_per_rev > 0.0
        }));

        let finer = evaluate_prescribed_helical_wake(
            &model,
            &bem(),
            PrescribedHelicalWakeOptions {
                wake_turns: 2,
                segments_per_turn: 24,
                ..PrescribedHelicalWakeOptions::default()
            },
        )
        .expect("finer wake");
        assert_ne!(result.segment_count, finer.segment_count);
        assert!(finer.maximum_induced_speed_m_s.is_finite());

        let faster_convection = evaluate_prescribed_helical_wake(
            &model,
            &bem(),
            PrescribedHelicalWakeOptions {
                axial_convection_factor: 1.5,
                ..PrescribedHelicalWakeOptions::default()
            },
        )
        .expect("scaled axial convection wake");
        assert!(
            faster_convection.stations[0].wake_pitch_m_per_rev
                > finer.stations[0].wake_pitch_m_per_rev
        );
    }

    #[test]
    fn prescribed_wake_rejects_invalid_options_and_source_state() {
        let model = model();
        let result = bem();
        for options in [
            PrescribedHelicalWakeOptions {
                wake_turns: 0,
                ..PrescribedHelicalWakeOptions::default()
            },
            PrescribedHelicalWakeOptions {
                segments_per_turn: 1,
                ..PrescribedHelicalWakeOptions::default()
            },
            PrescribedHelicalWakeOptions {
                core_radius_m: 0.0,
                ..PrescribedHelicalWakeOptions::default()
            },
            PrescribedHelicalWakeOptions {
                rotation_direction: 0.5,
                ..PrescribedHelicalWakeOptions::default()
            },
            PrescribedHelicalWakeOptions {
                max_segment_count: 0,
                ..PrescribedHelicalWakeOptions::default()
            },
            PrescribedHelicalWakeOptions {
                max_biot_savart_evaluations: 0,
                ..PrescribedHelicalWakeOptions::default()
            },
        ] {
            assert!(matches!(
                evaluate_prescribed_helical_wake(&model, &result, options),
                Err(PrescribedHelicalWakeError::InvalidOptions { .. })
            ));
        }

        assert!(matches!(
            evaluate_prescribed_helical_wake(
                &model,
                &result,
                PrescribedHelicalWakeOptions {
                    max_segment_count: 1,
                    ..PrescribedHelicalWakeOptions::default()
                },
            ),
            Err(PrescribedHelicalWakeError::InvalidOptions {
                field: "segment_count",
                ..
            })
        ));
        assert!(matches!(
            evaluate_prescribed_helical_wake(
                &model,
                &result,
                PrescribedHelicalWakeOptions {
                    max_biot_savart_evaluations: 1,
                    ..PrescribedHelicalWakeOptions::default()
                },
            ),
            Err(PrescribedHelicalWakeError::InvalidOptions {
                field: "biot_savart_evaluations",
                ..
            })
        ));

        let mut unconverged = result.clone();
        unconverged.converged = false;
        assert!(matches!(
            evaluate_prescribed_helical_wake(
                &model,
                &unconverged,
                PrescribedHelicalWakeOptions::default(),
            ),
            Err(PrescribedHelicalWakeError::InvalidBemResult { field: "converged" })
        ));

        let mut malformed = result;
        malformed.strips.pop();
        assert!(matches!(
            evaluate_prescribed_helical_wake(
                &model,
                &malformed,
                PrescribedHelicalWakeOptions::default(),
            ),
            Err(PrescribedHelicalWakeError::InvalidBemResult {
                field: "strip_diagnostics"
            })
        ));

        let mut invalid_diagnostics = bem();
        invalid_diagnostics.strips[0].residual = f64::NAN;
        assert!(matches!(
            evaluate_prescribed_helical_wake(
                &model,
                &invalid_diagnostics,
                PrescribedHelicalWakeOptions::default(),
            ),
            Err(PrescribedHelicalWakeError::InvalidBemResult {
                field: "induction_diagnostics"
            })
        ));
    }

    #[test]
    fn finite_vortex_segment_regularization_handles_degenerate_geometry() {
        let receiver = Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let contribution =
            finite_vortex_segment_velocity(receiver, receiver, receiver, 1.0, 1.0e-3);
        assert_eq!(contribution.x, 0.0);
        assert_eq!(contribution.y, 0.0);
        assert_eq!(contribution.z, 0.0);

        let nonzero = finite_vortex_segment_velocity(
            receiver,
            Point3 {
                x: -0.1,
                y: 0.0,
                z: 0.1,
            },
            Point3 {
                x: 0.1,
                y: 0.0,
                z: 0.1,
            },
            1.0,
            1.0e-3,
        );
        assert!(is_finite_point(nonzero));
        assert!(norm(nonzero) > 0.0);
    }
}
