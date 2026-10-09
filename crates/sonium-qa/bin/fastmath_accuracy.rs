//! Deterministic CPU QA for the public fastmath slice-dispatch accuracy API.
//!
//! This is a QA adapter rather than a second implementation of the accuracy
//! harness.  It emits the full per-regime `AccuracyReport` JSON so CI captures
//! architecture/compiler details alongside the release-facing QA record.

use sonium_fastmath::{AccuracyReport, CODY_WAITE_LIMIT, accuracy_report, reset_slow_path_count};
use sonium_qa::QaResult;

const CASE_ID: &str = "fastmath.accuracy.v2";
const MAX_ERROR_EPS: u64 = 8;
const MAX_P95_ERROR_EPS: u64 = 2;

fn main() {
    match run() {
        Ok(reports) => {
            let summary = Summary::from_reports(&reports);
            let reports_json = reports_json(&reports);
            println!("FASTMATH_ACCURACY_REPORT: {reports_json}");
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_sin_error_eps", summary.max_sin_error_eps as f64)
                .metric("max_cos_error_eps", summary.max_cos_error_eps as f64)
                .metric(
                    "max_p95_sin_error_eps",
                    summary.max_p95_sin_error_eps as f64,
                )
                .metric(
                    "max_p95_cos_error_eps",
                    summary.max_p95_cos_error_eps as f64,
                )
                .metric("max_sin_result_ulp", summary.max_sin_result_ulp as f64)
                .metric("max_cos_result_ulp", summary.max_cos_result_ulp as f64)
                .metric(
                    "max_p95_sin_result_ulp",
                    summary.max_p95_sin_result_ulp as f64,
                )
                .metric(
                    "max_p95_cos_result_ulp",
                    summary.max_p95_cos_result_ulp as f64,
                )
                .metric(
                    "min_sin_result_ulp_samples",
                    summary.min_sin_result_ulp_samples as f64,
                )
                .metric(
                    "min_cos_result_ulp_samples",
                    summary.min_cos_result_ulp_samples as f64,
                )
                .metric("max_sin_abs_error", summary.max_sin_abs_error)
                .metric("max_cos_abs_error", summary.max_cos_abs_error)
                .metric("total_samples", summary.total_samples as f64)
                .metric("slow_path_samples", summary.slow_path_samples as f64)
                .maximum("max_sin_error_eps", MAX_ERROR_EPS as f64)
                .maximum("max_cos_error_eps", MAX_ERROR_EPS as f64)
                .maximum("max_p95_sin_error_eps", MAX_P95_ERROR_EPS as f64)
                .maximum("max_p95_cos_error_eps", MAX_P95_ERROR_EPS as f64)
                .seed("fastmath-accuracy:deterministic-regime-union:v1");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("fastmath accuracy QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("fastmath accuracy QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Summary {
    max_sin_error_eps: u64,
    max_cos_error_eps: u64,
    max_p95_sin_error_eps: u64,
    max_p95_cos_error_eps: u64,
    max_sin_result_ulp: u64,
    max_cos_result_ulp: u64,
    max_p95_sin_result_ulp: u64,
    max_p95_cos_result_ulp: u64,
    min_sin_result_ulp_samples: usize,
    min_cos_result_ulp_samples: usize,
    max_sin_abs_error: f64,
    max_cos_abs_error: f64,
    total_samples: usize,
    slow_path_samples: u64,
}

impl Summary {
    fn from_reports(reports: &[(&str, AccuracyReport)]) -> Self {
        reports.iter().fold(
            Self {
                max_sin_error_eps: 0,
                max_cos_error_eps: 0,
                max_p95_sin_error_eps: 0,
                max_p95_cos_error_eps: 0,
                max_sin_result_ulp: 0,
                max_cos_result_ulp: 0,
                max_p95_sin_result_ulp: 0,
                max_p95_cos_result_ulp: 0,
                min_sin_result_ulp_samples: usize::MAX,
                min_cos_result_ulp_samples: usize::MAX,
                max_sin_abs_error: 0.0,
                max_cos_abs_error: 0.0,
                total_samples: 0,
                slow_path_samples: 0,
            },
            |mut summary, (_, report)| {
                summary.max_sin_error_eps = summary.max_sin_error_eps.max(report.max_sin_error_eps);
                summary.max_cos_error_eps = summary.max_cos_error_eps.max(report.max_cos_error_eps);
                summary.max_p95_sin_error_eps =
                    summary.max_p95_sin_error_eps.max(report.p95_sin_error_eps);
                summary.max_p95_cos_error_eps =
                    summary.max_p95_cos_error_eps.max(report.p95_cos_error_eps);
                summary.max_sin_result_ulp =
                    summary.max_sin_result_ulp.max(report.max_sin_result_ulp);
                summary.max_cos_result_ulp =
                    summary.max_cos_result_ulp.max(report.max_cos_result_ulp);
                summary.max_p95_sin_result_ulp = summary
                    .max_p95_sin_result_ulp
                    .max(report.p95_sin_result_ulp);
                summary.max_p95_cos_result_ulp = summary
                    .max_p95_cos_result_ulp
                    .max(report.p95_cos_result_ulp);
                summary.min_sin_result_ulp_samples = summary
                    .min_sin_result_ulp_samples
                    .min(report.sin_result_ulp_samples);
                summary.min_cos_result_ulp_samples = summary
                    .min_cos_result_ulp_samples
                    .min(report.cos_result_ulp_samples);
                summary.max_sin_abs_error = summary.max_sin_abs_error.max(report.max_sin_abs_error);
                summary.max_cos_abs_error = summary.max_cos_abs_error.max(report.max_cos_abs_error);
                summary.total_samples += report.samples;
                summary.slow_path_samples += report.slow_path_calls;
                summary
            },
        )
    }
}

fn run() -> Result<Vec<(&'static str, AccuracyReport)>, String> {
    let regimes = input_regimes();
    reset_slow_path_count();
    let reports: Vec<_> = regimes
        .iter()
        .map(|(name, input)| (*name, accuracy_report(input)))
        .collect();
    let summary = Summary::from_reports(&reports);

    for (name, value, threshold) in [
        (
            "max sin EPSILON-at-one error",
            summary.max_sin_error_eps,
            MAX_ERROR_EPS,
        ),
        (
            "max cos EPSILON-at-one error",
            summary.max_cos_error_eps,
            MAX_ERROR_EPS,
        ),
        (
            "max p95 sin EPSILON-at-one error",
            summary.max_p95_sin_error_eps,
            MAX_P95_ERROR_EPS,
        ),
        (
            "max p95 cos EPSILON-at-one error",
            summary.max_p95_cos_error_eps,
            MAX_P95_ERROR_EPS,
        ),
    ] {
        if value > threshold {
            return Err(format!(
                "{name} {value} exceeds published envelope {threshold}"
            ));
        }
    }
    if !summary.max_sin_abs_error.is_finite() || !summary.max_cos_abs_error.is_finite() {
        return Err("finite phase regimes produced a non-finite absolute error".into());
    }
    Ok(reports)
}

fn input_regimes() -> Vec<(&'static str, Vec<f64>)> {
    let small_reduced = (0..2048)
        .map(|index| (index as f64 - 1024.0) * 0.000_97)
        .collect();
    let coherent_phase_ramp = (0..4096)
        .map(|index| (index as f64 - 2048.0) * 0.037)
        .collect();
    let mut reduction_boundaries = Vec::with_capacity(33 * 64);
    for quadrant in -16..=16 {
        let center = quadrant as f64 * std::f64::consts::FRAC_PI_2;
        for offset in 0..64 {
            reduction_boundaries.push(center + (offset as f64 - 32.0) * 1.5e-13);
        }
    }
    let large_argument = (0..4096)
        .map(|index| {
            let magnitude = 10_f64.powf(3.0 * index as f64 / 4095.0);
            if index % 2 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect();
    let huge_fallback = vec![
        CODY_WAITE_LIMIT + 1.0,
        -CODY_WAITE_LIMIT - 1.0,
        1.0e12,
        -1.0e12,
        1.0e20,
        -1.0e20,
        f64::MAX,
        -f64::MAX,
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    vec![
        ("small_reduced", small_reduced),
        ("coherent_phase_ramp", coherent_phase_ramp),
        ("reduction_boundaries", reduction_boundaries),
        ("large_argument", large_argument),
        ("huge_fallback", huge_fallback),
    ]
}

fn reports_json(reports: &[(&str, AccuracyReport)]) -> String {
    let reports = reports
        .iter()
        .map(|(name, report)| format!("{{\"regime\":\"{name}\",\"report\":{}}}", report.to_json()))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"reports\":[{reports}]}}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_regimes_cover_fast_and_fallback_paths() {
        let reports = run().expect("deterministic accuracy regimes must meet the envelope");
        assert!(reports.iter().all(|(_, report)| report.samples > 0));
        assert!(
            reports
                .iter()
                .find(|(name, _)| *name == "huge_fallback")
                .is_some_and(|(_, report)| report.slow_path_calls >= 8)
        );
        assert!(reports_json(&reports).contains("reduction_boundaries"));
    }
}
