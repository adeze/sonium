//! Independent CPU evidence for seeded/scaled active-bound KKT gates.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{Bound, LbfgsbConfig, lbfgsb};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-scaled-kkt.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:solvers-optimization-scaled-kkt-v1";

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let mut cases = vec![
        (0.25_f64, 4.0_f64, [1.5_f64, -1.5_f64]),
        (1.0, 1.0, [1.25, -1.0]),
        (4.0, 0.25, [2.0, -2.0]),
        (16.0, 0.0625, [1.75, -1.25]),
    ];
    let mut seed = 0x8f3a_2d19_7c55_1041_u64;
    for _ in 0..8 {
        let scale0 = 0.125 + 15.875 * next_unit(&mut seed);
        let scale1 = 0.125 + 15.875 * next_unit(&mut seed);
        let target0 = 0.75 + 2.0 * next_unit(&mut seed);
        let target1 = -0.75 - 2.0 * next_unit(&mut seed);
        cases.push((scale0, scale1, [target0, target1]));
    }
    let mut max_kkt_norm = 0.0_f64;
    let mut max_stationarity = 0.0_f64;
    let mut seeded_max_kkt_norm = 0.0_f64;
    let mut seeded_max_stationarity = 0.0_f64;
    let mut seeded_case_count = 0.0_f64;
    let mut active_upper_cases = 0.0_f64;
    let mut active_lower_cases = 0.0_f64;

    for (index, (scale0, scale1, target)) in cases.iter().copied().enumerate() {
        let objective =
            |x: &[f64]| scale0 * (x[0] - target[0]).powi(2) + scale1 * (x[1] - target[1]).powi(2);
        let gradient = |x: &[f64]| {
            vec![
                2.0 * scale0 * (x[0] - target[0]),
                2.0 * scale1 * (x[1] - target[1]),
            ]
        };
        let config = LbfgsbConfig {
            bounds: vec![Bound::upper(0.5), Bound::lower(-0.25)],
            max_iterations: 200,
            grad_tolerance: 1.0e-9,
            ..LbfgsbConfig::default()
        };
        let result = lbfgsb(&objective, &gradient, &[0.0, 0.0], &config)
            .map_err(|error| format!("scaled L-BFGS-B failed: {error}"))?;
        if !result.converged {
            return Err(format!("scaled L-BFGS-B did not converge: {result:?}"));
        }
        if result.x.len() != 2
            || (result.x[0] - 0.5).abs() > 2.0e-5
            || (result.x[1] + 0.25).abs() > 2.0e-5
        {
            return Err(format!("active-bound solution mismatch: {:?}", result.x));
        }
        let report = result
            .kkt
            .as_ref()
            .ok_or_else(|| "successful scaled solve has no KKT report".to_string())?;
        if report.diagnostics.active_upper_bounds != [0]
            || report.diagnostics.active_lower_bounds != [1]
            || !report.norm.is_finite()
            || report.norm > 2.0e-5
            || report.multipliers.upper_bounds[0] <= 1.0e-3
            || report.multipliers.lower_bounds[1] <= 1.0e-3
        {
            return Err(format!("scaled KKT gate failed: {report:?}"));
        }
        let g = gradient(&result.x);
        let stationarity = (g[0] + report.multipliers.upper_bounds[0])
            .abs()
            .max((g[1] - report.multipliers.lower_bounds[1]).abs());
        if !stationarity.is_finite() || stationarity > 2.0e-5 {
            return Err(format!("scaled stationarity mismatch: {stationarity:e}"));
        }
        max_kkt_norm = max_kkt_norm.max(report.norm);
        max_stationarity = max_stationarity.max(stationarity);
        if index >= 4 {
            seeded_case_count += 1.0;
            seeded_max_kkt_norm = seeded_max_kkt_norm.max(report.norm);
            seeded_max_stationarity = seeded_max_stationarity.max(stationarity);
        }
        active_upper_cases += 1.0;
        active_lower_cases += 1.0;
    }

    Ok(vec![
        ("scaled_case_count", cases.len() as f64),
        ("active_upper_cases", active_upper_cases),
        ("active_lower_cases", active_lower_cases),
        ("max_kkt_norm", max_kkt_norm),
        ("max_stationarity", max_stationarity),
        ("seeded_case_count", seeded_case_count),
        ("seeded_max_kkt_norm", seeded_max_kkt_norm),
        ("seeded_max_stationarity", seeded_max_stationarity),
    ])
}

fn next_unit(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference_checksum(CHECKSUM)
                .minimum("scaled_case_count", 12.0)
                .minimum("seeded_case_count", 8.0)
                .minimum("active_upper_cases", 12.0)
                .minimum("active_lower_cases", 12.0)
                .maximum("max_kkt_norm", 3.0e-5)
                .maximum("max_stationarity", 3.0e-5)
                .maximum("seeded_max_kkt_norm", 3.0e-5)
                .maximum("seeded_max_stationarity", 3.0e-5);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let mut result = QaResult::failed(CASE_ID, error).reference_checksum(CHECKSUM);
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
