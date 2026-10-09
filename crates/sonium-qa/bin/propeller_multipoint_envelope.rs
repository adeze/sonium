//! Multipoint propeller operating-condition reference QA.

use std::f64::consts::PI;
use std::time::Instant;

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_qa::QaResult;
use sonium_sim_propellers::{
    AIR_GAMMA, AIR_GAS_CONSTANT_J_KG_K, Blade3dConfig, Propeller3dOptConfig,
    PropellerOperatingCondition, PropellerOperatingEnvelope, try_optimize_blade_shape,
};

const CASE_ID: &str = "propeller.multipoint-operating-envelope.v1";

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

fn run() -> Result<(usize, f64, f64, f64, usize, f64), String> {
    let source = PropellerSource::drone_5inch(18_000.0);
    let envelope = PropellerOperatingEnvelope::try_from_source(source.clone())
        .map_err(|error| error.to_string())?;
    let source_temperature_k =
        source.speed_of_sound.powi(2) / (AIR_GAMMA * AIR_GAS_CONSTANT_J_KG_K);
    let conditions = [
        PropellerOperatingCondition {
            freestream_m_s: 0.0,
            rpm: source.rpm,
            inflow_angle_rad: 0.0,
            density_kg_m3: source.density,
            temperature_k: source_temperature_k,
        },
        PropellerOperatingCondition {
            freestream_m_s: 8.0,
            rpm: source.rpm,
            inflow_angle_rad: 0.35,
            density_kg_m3: source.density,
            temperature_k: source_temperature_k,
        },
        PropellerOperatingCondition {
            freestream_m_s: 3.0,
            rpm: source.rpm * 0.8,
            inflow_angle_rad: -0.2,
            density_kg_m3: source.density * 1.1,
            temperature_k: source_temperature_k * 0.9,
        },
        PropellerOperatingCondition {
            freestream_m_s: 8.0,
            rpm: source.rpm,
            inflow_angle_rad: -0.35,
            density_kg_m3: source.density,
            temperature_k: source_temperature_k * 1.15,
        },
    ];
    let points = envelope
        .evaluate_conditions(&conditions)
        .map_err(|error| error.to_string())?;
    if points.len() != conditions.len() {
        return Err("multipoint evaluator changed input cardinality".into());
    }

    let disk_area_m2 = envelope.disk_area_m2();
    let mut max_load_error = 0.0_f64;
    let mut max_induced_error = 0.0_f64;
    let mut max_frame_error = 0.0_f64;
    for (condition, point) in conditions.iter().zip(&points) {
        let rpm_ratio = condition.rpm / source.rpm;
        let load_scale = condition.density_kg_m3 / source.density * rpm_ratio.powi(2);
        let thrust_n = source.thrust * load_scale;
        let torque_nm = source.torque * load_scale;
        let shaft_power_w = torque_nm * 2.0 * PI * condition.rpm / 60.0;
        let axial_freestream_m_s = condition.freestream_m_s * condition.inflow_angle_rad.cos();
        let discriminant = axial_freestream_m_s.powi(2)
            + 2.0 * thrust_n / (condition.density_kg_m3 * disk_area_m2);
        let induced_velocity_m_s = thrust_n
            / (condition.density_kg_m3
                * disk_area_m2
                * (discriminant.sqrt() + axial_freestream_m_s));
        max_load_error = max_load_error.max(relative_error(point.load_scale, load_scale));
        max_load_error = max_load_error.max(relative_error(point.thrust_n, thrust_n));
        max_load_error = max_load_error.max(relative_error(point.torque_nm, torque_nm));
        max_load_error = max_load_error.max(relative_error(point.shaft_power_w, shaft_power_w));
        max_induced_error = max_induced_error.max(relative_error(
            point.induced_velocity_m_s,
            induced_velocity_m_s,
        ));
        max_frame_error =
            max_frame_error.max((point.axial_freestream_m_s - axial_freestream_m_s).abs());
        if point.power_margin_w < -1.0e-10
            || !point.speed_of_sound_m_s.is_finite()
            || point.speed_of_sound_m_s <= 0.0
        {
            return Err(format!("invalid admitted point: {point:?}"));
        }
    }
    let frame_error = (points[1].axial_freestream_m_s - points[3].axial_freestream_m_s).abs();
    max_frame_error = max_frame_error.max(frame_error);
    if max_load_error > 1.0e-12 || max_induced_error > 1.0e-12 || max_frame_error > 1.0e-12 {
        return Err(format!(
            "multipoint reference mismatch: load={max_load_error:e}, induced={max_induced_error:e}, frame={max_frame_error:e}"
        ));
    }

    let optimizer = try_optimize_blade_shape(&Propeller3dOptConfig {
        source: source.clone(),
        mesh: Blade3dConfig {
            radial_segments: 8,
            chord_segments: 3,
            loop_segments: 6,
            ..Blade3dConfig::default()
        },
        n_harmonics: 4,
        max_iterations: 1,
        operating_conditions: conditions.to_vec(),
        ..Propeller3dOptConfig::default()
    })
    .map_err(|error| format!("multipoint optimizer admission failed: {error}"))?;
    if !optimizer.is_feasible() || !optimizer.is_exportable() {
        return Err(format!(
            "multipoint optimizer result was not promotable: feasible={}, exportable={}",
            optimizer.is_feasible(),
            optimizer.is_exportable()
        ));
    }
    if optimizer.operating_points.len() != conditions.len() {
        return Err(format!(
            "multipoint optimizer changed input cardinality: {} != {}",
            optimizer.operating_points.len(),
            conditions.len()
        ));
    }
    let optimizer_reference = PropellerOperatingEnvelope::try_from_source(optimizer.design.source)
        .map_err(|error| format!("optimized source could not rebuild envelope: {error}"))?
        .evaluate_conditions(&conditions)
        .map_err(|error| format!("optimized source failed direct multipoint reference: {error}"))?;
    let mut max_optimizer_reference_error = 0.0_f64;
    for (expected, actual) in optimizer_reference.iter().zip(&optimizer.operating_points) {
        max_optimizer_reference_error = max_optimizer_reference_error
            .max(relative_error(actual.load_scale, expected.load_scale))
            .max(relative_error(
                actual.induced_velocity_m_s,
                expected.induced_velocity_m_s,
            ))
            .max(relative_error(actual.shaft_power_w, expected.shaft_power_w));
    }
    if max_optimizer_reference_error > 1.0e-12 {
        return Err(format!(
            "multipoint optimizer/reference mismatch: {max_optimizer_reference_error:e}"
        ));
    }

    Ok((
        points.len(),
        max_load_error,
        max_induced_error,
        max_frame_error,
        optimizer.operating_points.len(),
        max_optimizer_reference_error,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            point_count,
            max_load_error,
            max_induced_error,
            max_frame_error,
            optimizer_point_count,
            max_optimizer_reference_error,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("operating_point_count", point_count as f64)
                .metric("max_load_reference_relative_error", max_load_error)
                .metric("max_induced_velocity_reference_relative_error", max_induced_error)
                .metric("max_frame_transform_absolute_error", max_frame_error)
                .metric("optimizer_operating_point_count", optimizer_point_count as f64)
                .metric(
                    "max_optimizer_reference_relative_error",
                    max_optimizer_reference_error,
                )
                .input_checksum("propeller-5inch:18000rpm:multipoint-envelope-v1")
                .reference(
                    "independent constant-load-coefficient rho*RPM^2 scaling, ideal-gas sound speed, and annular momentum reference",
                )
                .reference_checksum("analytic/deterministic-fixture:propeller-multipoint-envelope-v1")
                .dimension("four conditions: hover, forward inflow, reduced RPM/density-scaled load, and symmetric inflow angles")
                .formulation("T,Q scale with density*RPM^2; sound speed=sqrt(gamma*R*T); disk solve uses axial V=|V|cos(angle)")
                .tolerance("all independent load and induced-velocity relative errors <= 1e-12; frame-transform absolute error <= 1e-12; non-negative power margin")
                .minimum("operating_point_count", 4.0)
                .maximum("max_load_reference_relative_error", 1.0e-12)
                .maximum("max_induced_velocity_reference_relative_error", 1.0e-12)
                .maximum("max_frame_transform_absolute_error", 1.0e-12)
                .minimum("optimizer_operating_point_count", 4.0)
                .maximum("max_optimizer_reference_relative_error", 1.0e-12)
                .seed("propeller-multipoint-envelope-v1:conditions=4:source=drone-5inch")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PROPELLER MULTIPOINT ENVELOPE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
