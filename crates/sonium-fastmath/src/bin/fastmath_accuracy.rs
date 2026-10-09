//! Emit a deterministic machine-readable fastmath accuracy artifact.
//!
//! The output is a single JSON object suitable for CI artifacts or a
//! benchmark harness. Samples cover small phases, reduction boundaries, and
//! the documented Cody-Waite cutoff.

use sonium_fastmath::{CODY_WAITE_LIMIT, accuracy_report, reset_slow_path_count};

fn report_json(regime: &str, samples: &[f64]) -> String {
    format!(
        "{{\"regime\":\"{regime}\",\"report\":{}}}",
        accuracy_report(samples).to_json()
    )
}

fn full_finite_domain_samples() -> Vec<f64> {
    let fractions = [0_u64, 1, 1_u64 << 51, (1_u64 << 52) - 1];
    let mut samples = Vec::with_capacity(0x7fe * fractions.len() * 2);
    for exponent in 1_u64..=0x7fe {
        for &fraction in &fractions {
            let bits = (exponent << 52) | fraction;
            samples.push(f64::from_bits(bits));
            samples.push(f64::from_bits(bits | (1_u64 << 63)));
        }
    }
    samples
}

fn main() {
    let mut small_reduced = Vec::with_capacity(2048);
    for i in 0..2048 {
        small_reduced.push((i as f64 - 1024.0) * 0.000_97);
    }

    let coherent_phase_ramp: Vec<f64> = (0..4096).map(|i| (i as f64 - 2048.0) * 0.037).collect();

    let mut reduction_boundaries = Vec::with_capacity(4096);
    for k in -64..=64 {
        let center = k as f64 * std::f64::consts::FRAC_PI_2;
        for j in 0..32 {
            reduction_boundaries.push(center + (j as f64 - 16.0) * 1.0e-12);
        }
    }

    let mut random_full_range = Vec::with_capacity(4096);
    let mut state = 0x5eed_u64;
    for _ in 0..4096 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = state as f64 / u64::MAX as f64;
        random_full_range.push((2.0 * unit - 1.0) * CODY_WAITE_LIMIT);
    }

    let huge_and_special = vec![
        CODY_WAITE_LIMIT + 1.0,
        -CODY_WAITE_LIMIT - 1.0,
        1.0e12,
        -1.0e12,
        1.0e20,
        -1.0e20,
        f64::MAX,
        -f64::MAX,
        -1_000.0,
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let full_finite_domain = full_finite_domain_samples();

    reset_slow_path_count();
    println!(
        "{{\"architecture\":\"{}\",\"reports\":[{}, {}, {}, {}, {}, {}]}}",
        std::env::consts::ARCH,
        report_json("small_reduced", &small_reduced),
        report_json("coherent_phase_ramp", &coherent_phase_ramp),
        report_json("reduction_boundaries", &reduction_boundaries),
        report_json("random_full_range", &random_full_range),
        report_json("huge_and_special", &huge_and_special),
        report_json("full_finite_domain", &full_finite_domain),
    );
}
