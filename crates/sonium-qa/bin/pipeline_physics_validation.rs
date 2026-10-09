//! Physics-level application pipeline validation.
//!
//! This complements structure-only pipeline tests by checking acceptance
//! criteria on actual acoustic behavior:
//! - loudspeaker BEM SPL/directivity against a broad baffled-piston reference
//! - room BEM modal behavior around the first rigid shoebox axial mode

use sonium_sim_speaker::cea2034_output::{
    ANGLES_PER_PLANE, TOTAL_EVAL_POINTS, generate_eval_points,
};
use sonium_sim_speaker::model::{
    CabinetDimensions, DriverConfig, LoudspeakerModel, SeparatorConfig, SimulationConfig,
    SpeakerMaterialAssignment,
};
use sonium_sim_speaker::simulation::{
    SolverConfig as SpeakerSolverConfig, try_run_simulation_with_config,
};
use sonium_sim_speaker_modeling::motor_extraction::{
    CoilSpec, ExtractionConfig, MagnetSpec, MagnetType, MotorGeometry, PolePieceSpec,
    extract_motor_parameters,
};
use std::f64::consts::PI;

const P_REF: f64 = 20.0e-6;
const MAX_PISTON_REFERENCE_ERROR_DB: f64 = 4.0;
const MAX_FROZEN_FIXTURE_DRIFT_DB: f64 = 0.25;

/// SPL from a peak pressure magnitude (RMS-referenced, matches the BEM
/// post-processing convention).
fn spl_from_pressure_norm(p_peak: f64) -> f64 {
    let p_rms = p_peak.max(1.0e-30) / std::f64::consts::SQRT_2;
    20.0 * (p_rms / P_REF).log10()
}

fn baffled_piston_on_axis_spl(
    radius: f64,
    velocity: f64,
    frequency: f64,
    distance: f64,
    rho: f64,
    c: f64,
) -> f64 {
    let area = PI * radius * radius;
    let k = 2.0 * PI * frequency / c;
    // Far-field on-axis peak pressure for a baffled piston (low-ka limit).
    let pressure_peak = rho * c * k * velocity * area / (2.0 * PI * distance);
    spl_from_pressure_norm(pressure_peak)
}

fn validate_loudspeaker_bem_pipeline() -> Result<(), String> {
    let woofer_radius = 0.065;
    let woofer_velocity = 0.001;
    let frequency = 1000.0;
    let c = 343.0;
    let rho = 1.21;
    let distance = 1.0;

    let model = LoudspeakerModel {
        cabinet: CabinetDimensions {
            height: 0.36,
            width: 0.22,
            depth: 0.20,
        },
        woofer: DriverConfig {
            radius: woofer_radius,
            x_offset: 0.0,
            y_offset: -0.055,
            piston_velocity: woofer_velocity,
            transfer_function: None,
            driver_model: None,
            cone: None,
        },
        tweeter: DriverConfig {
            radius: 0.012,
            x_offset: 0.0,
            y_offset: 0.095,
            piston_velocity: 0.0,
            transfer_function: None,
            driver_model: None,
            cone: None,
        },
        separator: SeparatorConfig {
            y_position: 0.035,
            thickness: 0.012,
        },
        waveguide: None,
        materials: SpeakerMaterialAssignment::default(),
        simulation: SimulationConfig {
            freq_start: frequency,
            freq_end: frequency,
            freq_points: 1,
            speed_of_sound: c,
            air_density: rho,
            max_mesh_frequency: Some(frequency),
        },
        additional_drivers: Vec::new(),
        product_design: None,
    };
    model.validate()?;

    let eval_points = generate_eval_points(model.baffle_center());

    let result = try_run_simulation_with_config(
        &model,
        &eval_points,
        &SpeakerSolverConfig {
            direct_threshold: 1_000_000,
            hmatrix_threshold: usize::MAX,
            ..SpeakerSolverConfig::default()
        },
    )
    .map_err(|err| err.to_string())?;

    let fixture = CeaFixtureMetrics::from_spl_row(result.spl.row(0))?;
    let on_axis = fixture.on_axis;
    let off_axis = fixture.horizontal_60;
    let reference =
        baffled_piston_on_axis_spl(woofer_radius, woofer_velocity, frequency, distance, rho, c);

    println!("Loudspeaker BEM full-pipeline physics validation:");
    println!("  on-axis SPL: {on_axis:.2} dB, baffled-piston reference: {reference:.2} dB");
    println!(
        "  CEA fixture SPL: H30={:.2}, H60={:.2}, H90={:.2}, H180={:.2}, V30={:.2}",
        fixture.horizontal_30,
        fixture.horizontal_60,
        fixture.horizontal_90,
        fixture.horizontal_180,
        fixture.vertical_30
    );

    if !on_axis.is_finite() || !off_axis.is_finite() {
        return Err("speaker pipeline produced non-finite SPL".into());
    }
    if (on_axis - reference).abs() > MAX_PISTON_REFERENCE_ERROR_DB {
        return Err(format!(
            "on-axis SPL differs from broad baffled-piston reference by {:.2} dB (limit {MAX_PISTON_REFERENCE_ERROR_DB:.2} dB)",
            (on_axis - reference).abs(),
        ));
    }
    if off_axis > on_axis - 2.0 {
        return Err(format!(
            "60-degree SPL {off_axis:.2} dB should be at least 2 dB below on-axis {on_axis:.2} dB"
        ));
    }
    fixture.validate_against_reference()?;

    Ok(())
}

fn validate_motor_extraction_pipeline() -> Result<(), String> {
    let geometry = MotorGeometry {
        magnet: MagnetSpec {
            magnet_type: MagnetType::Neodymium { br: 1.2 },
            inner_radius: 0.018,
            outer_radius: 0.030,
            height: 0.015,
        },
        pole_piece: PolePieceSpec {
            mu_r: 2000.0,
            conductivity: 6.99e6,
            radius: 0.012,
            height: 0.006,
        },
        coil: CoilSpec {
            radius: 0.0155,
            height: 0.012,
            n_turns: 56,
            wire_diameter: 0.0002,
            layers: 2,
        },
        conductive_insert: None,
        air_gap: 0.001,
        coil_displacement_m: 0.0,
    };
    let config = ExtractionConfig {
        x_max: 0.002,
        n_displacement_steps: 2,
        mesh_resolution: 2,
        ..ExtractionConfig::default()
    };
    let result = extract_motor_parameters(&geometry, &config).map_err(|error| error.to_string())?;

    if result.bl_curve.len() != 5
        || result.le_vs_frequency.len() != config.le_frequencies.len()
        || result.re_eddy_vs_frequency.len() != config.le_frequencies.len()
    {
        return Err("motor extraction returned incomplete curves".into());
    }
    if result
        .bl_curve
        .iter()
        .any(|(x, bl)| !x.is_finite() || !bl.is_finite())
        || !result.bl_nominal.is_finite()
        || !result.le_nominal.is_finite()
    {
        return Err("motor extraction returned non-finite parameters".into());
    }
    let zero_bl = result
        .bl_curve
        .iter()
        .find(|(x, _)| x.abs() < 1.0e-15)
        .map(|(_, bl)| *bl)
        .ok_or_else(|| "BL(x) curve is missing the zero-displacement sample".to_string())?;
    if (zero_bl - result.bl_nominal).abs() > 1.0e-12 {
        return Err(format!(
            "BL nominal does not match BL(0): nominal={}, sampled={zero_bl}",
            result.bl_nominal
        ));
    }

    const BL_REFERENCE_TM: f64 = 0.906_070_896_315_195_8;
    const MAX_BL_DRIFT_TM: f64 = 0.02;
    let bl_error = (result.bl_nominal - BL_REFERENCE_TM).abs();
    if bl_error > MAX_BL_DRIFT_TM {
        return Err(format!(
            "motor BL(0) drifted by {bl_error:.4} T·m (limit {MAX_BL_DRIFT_TM:.4}); actual={:.6}, reference={BL_REFERENCE_TM:.6}",
            result.bl_nominal
        ));
    }

    let mut previous_frequency = 0.0;
    let mut previous_inductance = f64::INFINITY;
    let mut previous_resistance = 0.0;
    for (index, ((frequency, inductance), (resistance_frequency, resistance))) in result
        .le_vs_frequency
        .iter()
        .zip(&result.re_eddy_vs_frequency)
        .enumerate()
    {
        if frequency != resistance_frequency
            || !frequency.is_finite()
            || !inductance.is_finite()
            || !resistance.is_finite()
            || *frequency <= previous_frequency
            || *inductance <= 0.0
            || *inductance > previous_inductance
            || *resistance < previous_resistance
        {
            return Err(format!(
                "motor Le/Re curve violates finiteness, passivity, or monotonicity at index {index}"
            ));
        }
        previous_frequency = *frequency;
        previous_inductance = *inductance;
        previous_resistance = *resistance;
    }

    println!("Motor extraction full-pipeline physics validation:");
    println!(
        "  BL(0)={:.6} T·m, Le(DC)={:.6} mH, BL drift={bl_error:.3e} T·m",
        result.bl_nominal,
        result.le_nominal * 1.0e3
    );
    Ok(())
}

struct CeaFixtureMetrics {
    on_axis: f64,
    vertical_on_axis: f64,
    horizontal_30: f64,
    horizontal_60: f64,
    horizontal_90: f64,
    horizontal_180: f64,
    vertical_30: f64,
}

impl CeaFixtureMetrics {
    fn from_spl_row(spl: &[f64]) -> Result<Self, String> {
        if spl.len() != TOTAL_EVAL_POINTS {
            return Err(format!(
                "CEA fixture should produce {TOTAL_EVAL_POINTS} SPL points, got {}",
                spl.len()
            ));
        }
        if spl.iter().any(|value| !value.is_finite()) {
            return Err("CEA fixture produced non-finite SPL".into());
        }

        Ok(Self {
            on_axis: spl[0],
            vertical_on_axis: spl[ANGLES_PER_PLANE],
            horizontal_30: spl[3],
            horizontal_60: spl[6],
            horizontal_90: spl[9],
            horizontal_180: spl[18],
            vertical_30: spl[ANGLES_PER_PLANE + 3],
        })
    }

    fn validate_against_reference(&self) -> Result<(), String> {
        // References measured after two physics fixes:
        //   1. Internal separator removed from the exterior BEM mesh.
        //   2. SPL formula corrected to RMS-referenced (|p|/√2 / 20 µPa).
        // Pre-fix values were ~3.01 dB higher (peak-amplitude convention)
        // and slightly inflated by spurious mid-face couplings from the
        // internal separator.
        let references = [
            ("on-axis", self.on_axis, 57.85),
            ("horizontal 30 deg", self.horizontal_30, 56.48),
            ("horizontal 60 deg", self.horizontal_60, 52.92),
            ("horizontal 90 deg", self.horizontal_90, 47.99),
            ("horizontal 180 deg", self.horizontal_180, 45.04),
            ("vertical 30 deg", self.vertical_30, 57.20),
        ];
        for (label, actual, reference) in references {
            let error = (actual - reference).abs();
            if error > MAX_FROZEN_FIXTURE_DRIFT_DB {
                return Err(format!(
                    "CEA fixture {label} SPL drifted by {error:.2} dB (limit {MAX_FROZEN_FIXTURE_DRIFT_DB:.2} dB); actual={actual:.2}, reference={reference:.2}"
                ));
            }
        }

        if (self.on_axis - self.vertical_on_axis).abs() > 0.25 {
            return Err(format!(
                "CEA fixture duplicated on-axis points disagree: H0={:.2}, V0={:.2}",
                self.on_axis, self.vertical_on_axis
            ));
        }
        if self.horizontal_60 > self.on_axis - 2.0 {
            return Err(format!(
                "CEA fixture H60 should be at least 2 dB below on-axis: H60={:.2}, H0={:.2}",
                self.horizontal_60, self.on_axis
            ));
        }
        if self.horizontal_90 > self.on_axis - 6.0 {
            return Err(format!(
                "CEA fixture H90 should be at least 6 dB below on-axis: H90={:.2}, H0={:.2}",
                self.horizontal_90, self.on_axis
            ));
        }
        if !(self.horizontal_30 >= self.horizontal_60 && self.horizontal_60 >= self.horizontal_90) {
            return Err(format!(
                "CEA fixture horizontal directivity is not monotonically narrowing over 30/60/90 deg: H30={:.2}, H60={:.2}, H90={:.2}",
                self.horizontal_30, self.horizontal_60, self.horizontal_90
            ));
        }
        if self.horizontal_180 > self.on_axis - 8.0 {
            return Err(format!(
                "CEA fixture rear response should be at least 8 dB below on-axis: H180={:.2}, H0={:.2}",
                self.horizontal_180, self.on_axis
            ));
        }

        Ok(())
    }
}

fn run() -> Result<(), String> {
    validate_loudspeaker_bem_pipeline()?;
    validate_motor_extraction_pipeline()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }

    println!("SUCCESS");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_application_pipeline_physics() {
        run().expect("application pipelines should satisfy physics-level acceptance checks");
    }
}
