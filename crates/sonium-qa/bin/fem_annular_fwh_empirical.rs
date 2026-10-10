//! Annular FW-H vs Gutin empirical tones on the PC-fan product annulus.
//!
//! Large-scale validation T3: the distributed spinning-dipole FW-H layer
//! ([`sonium_sim_pcfan::annular_loading_tones`]) over the real product
//! hub/tip annulus must reproduce the compact rotating-dipole Gutin model
//! ([`PropellerSource`]) in the compact limit. Both models radiate through
//! the same Bessel factor `J_{hB}(hB·M_t·sinθ)` (spinning-mode radiation);
//! the harness normalizes out the known finite-blade-width factor
//! `sinc(hπσ)` that only the distributed model carries, then compares the
//! underlying radiation physics.
//!
//! Gated cases use few-blade test rotors (`B = 2, 3` at 6000 rpm) where the
//! tones sit well above the hoop-quadrature floor. The eight-blade product
//! point (600 rpm, `M_t ≈ 0.01`) is `J_8`-suppressed by ~15 orders of
//! magnitude — below any quadrature floor — so it is gated as a
//! suppression bound instead of an equality.
//!
//! The gated ratio divides out three analytic differences between the
//! models, each confirmed independently by the four-case spread:
//!
//! 1. blade-width factor `sinc(hπσ)` (distributed model only),
//! 2. radial-distribution factor `2/(m+2)·(1−q^{m+2})/(1−q²)` with hub/tip
//!    ratio `q` (`J_m ∝ r^m` integrated over the annulus vs concentrated
//!    at the tip; varies 0.28–0.56 across the cases),
//! 3. axial-dipole obliquity `cosθ` (caller-side per the empirical model's
//!    contract; uniform 0.707 at the 45° observer).
//!
//! What remains is the shared spinning-dipole radiation physics, which
//! agrees to ~3% (residual: near-field `1/kr` at `kr ≈ 11–22`).

use sonium_fem::physics::acoustics::aeroacoustics::{PropellerSource, RevolvedQuadrature};
use sonium_sim_pcfan::{AnnularToneSpec, PcFanConfig, annular_loading_tones};
use std::f64::consts::PI;

/// Steady thrust shared by both models [N].
const THRUST_N: f64 = 2.0;
/// Blade solidity shared by the FW-H strips.
const SOLIDITY: f64 = 0.35;
/// Observer radius [m] (kr ≈ 11–22: far enough that the empirical
/// far-field form applies to ~10%).
const OBSERVER_R_M: f64 = 3.0;
/// Observer polar angle from the rotor axis [rad].
const OBSERVER_THETA: f64 = PI / 4.0;

struct QaCheck {
    method: &'static str,
    name: String,
    metric: f64,
    maximum: f64,
}

impl QaCheck {
    fn passed(&self) -> bool {
        self.metric <= self.maximum
    }
}

fn sinc(argument: f64) -> f64 {
    argument.sin() / argument
}

fn run_equality_case(blade_count: usize, rpm: f64, harmonic: usize) -> Vec<QaCheck> {
    let config = PcFanConfig::default();
    let hub = config.dimensions.hub_radius_m;
    let tip = config.dimensions.blade_tip_radius_m;
    let density = config.operating_point.air_density_kg_m3;
    let sound_speed = config.operating_point.speed_of_sound_m_s;

    let spec = AnnularToneSpec {
        blade_count,
        rpm,
        thrust_n: THRUST_N,
        hub_radius_m: hub,
        tip_radius_m: tip,
        blade_solidity: SOLIDITY,
        strip_count: 8,
        harmonics: vec![harmonic],
        density_kg_m3: density,
        sound_speed_m_s: sound_speed,
    };
    let observer = [
        OBSERVER_R_M * OBSERVER_THETA.sin(),
        0.0,
        OBSERVER_R_M * OBSERVER_THETA.cos(),
    ];
    let quadrature = RevolvedQuadrature::new(1e-10, 12).expect("quadrature controls");
    let tones = annular_loading_tones(&spec, observer, quadrature).expect("annular tones");
    let tone = tones[0];

    let empirical = PropellerSource {
        n_blades: blade_count,
        rpm,
        thrust: THRUST_N,
        torque: 0.0,
        tip_radius: tip,
        hub_radius: hub,
        speed_of_sound: sound_speed,
        density,
    };
    let expected_frequency = empirical.harmonic_frequency(harmonic);
    let expected_level =
        empirical.harmonic_amplitude_at_direction(harmonic, OBSERVER_R_M, OBSERVER_THETA);

    // The distributed model carries the blade-width factor sinc(hπσ); the
    // compact model does not. Divide it out before comparing radiation.
    let width_factor = sinc(harmonic as f64 * PI * SOLIDITY);
    // Radial distribution: J_m ∝ r^m integrated uniformly over the annulus
    // vs concentrated at the tip in the compact model.
    let order = (harmonic * blade_count) as f64;
    let hub_tip_ratio = hub / tip;
    let radial_factor = (2.0 / (order + 2.0)) * (1.0 - hub_tip_ratio.powf(order + 2.0))
        / (1.0 - hub_tip_ratio.powi(2));
    // Axial-dipole obliquity, caller-side per the empirical model's contract.
    let obliquity = OBSERVER_THETA.cos();
    let fwh_level = tone.pressure_pa.norm() / (width_factor * radial_factor * obliquity);
    let relative = (fwh_level - expected_level).abs() / expected_level;

    println!(
        "B={blade_count} h={harmonic} m={} f={:.1} Hz |FW-H|={:.4e} Pa sinc={:.4} radial={:.4} |FW-H|/corr={:.4e} Pa empirical={:.4e} Pa rel={:.4e}",
        tone.azimuthal_order,
        tone.frequency_hz,
        tone.pressure_pa.norm(),
        width_factor,
        radial_factor,
        fwh_level,
        expected_level,
        relative,
    );

    vec![
        QaCheck {
            method: "FW-H",
            name: format!("B={blade_count} h={harmonic} tone frequency"),
            metric: (tone.frequency_hz - expected_frequency).abs() / expected_frequency,
            maximum: 1e-12,
        },
        QaCheck {
            method: "FW-H",
            name: format!("B={blade_count} h={harmonic} level vs Gutin"),
            metric: relative,
            maximum: 1e-1,
        },
    ]
}

fn run_product_suppression_case() -> Vec<QaCheck> {
    let config = PcFanConfig::default();
    let spec = AnnularToneSpec {
        blade_count: config.blade_count,
        rpm: config.operating_point.rpm,
        thrust_n: THRUST_N,
        hub_radius_m: config.dimensions.hub_radius_m,
        tip_radius_m: config.dimensions.blade_tip_radius_m,
        blade_solidity: SOLIDITY,
        strip_count: 8,
        harmonics: vec![1],
        density_kg_m3: config.operating_point.air_density_kg_m3,
        sound_speed_m_s: config.operating_point.speed_of_sound_m_s,
    };
    let observer = [
        OBSERVER_R_M * OBSERVER_THETA.sin(),
        0.0,
        OBSERVER_R_M * OBSERVER_THETA.cos(),
    ];
    let quadrature = RevolvedQuadrature::new(1e-10, 12).expect("quadrature controls");
    let tones = annular_loading_tones(&spec, observer, quadrature).expect("annular tones");
    let level = tones[0].pressure_pa.norm();
    println!(
        "product point B={} rpm={:.0} h=1 f={:.1} Hz |FW-H|={:.4e} Pa (J_8-suppressed)",
        config.blade_count, config.operating_point.rpm, tones[0].frequency_hz, level,
    );
    // Unsuppressed compact level would be ~8e-2 Pa; the spinning
    // cancellation must hold the tone below 1e-6 Pa (>90 dB down).
    vec![QaCheck {
        method: "FW-H",
        name: "product point BPF suppression bound".to_string(),
        metric: level,
        maximum: 1e-6,
    }]
}

fn main() {
    let mut checks = Vec::new();
    for blade_count in [2, 3] {
        for harmonic in [1, 2] {
            checks.extend(run_equality_case(blade_count, 6_000.0, harmonic));
        }
    }
    checks.extend(run_product_suppression_case());

    println!("Annular FW-H vs Gutin empirical validation");
    println!(
        "{:<12} {:<48} {:>12} {:>12}  result",
        "method", "check", "metric", "maximum"
    );
    for result in &checks {
        println!(
            "{:<12} {:<48} {:>12.4e} {:>12.4e}  {}",
            result.method,
            result.name,
            result.metric,
            result.maximum,
            if result.passed() { "pass" } else { "fail" }
        );
    }
    let failures: Vec<_> = checks.iter().filter(|result| !result.passed()).collect();
    if failures.is_empty() {
        println!("SUCCESS");
    } else {
        eprintln!("FAILED: {} of {} QA checks", failures.len(), checks.len());
        std::process::exit(1);
    }
}
