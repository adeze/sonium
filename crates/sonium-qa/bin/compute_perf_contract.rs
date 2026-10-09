//! CPU-only roofline accounting contract for sonium-compute.
//!
//! This intentionally does not benchmark or launch a simulator.  It checks
//! the independent byte/FLOP formulas used by the Criterion benches and
//! verifies that extreme caller-controlled shapes saturate instead of
//! wrapping or panicking.

use sonium_compute::CsrRowBucketStats;
use sonium_compute::perf::{
    PerfModel, axpy_model, bem_regular_pair_c32_model, cast_model, copy_d2d_model, copy_h2d_model,
    csr_spmv_model, dot_model, gather4_weighted_model, gemm_model, gemv_model, nrm2_model,
    scale_model,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.perf-contract.v1";

#[derive(Debug)]
struct Metrics {
    case_count: usize,
    saturation_cases: usize,
    normal_cases: usize,
}

fn expect(model: PerfModel, bytes: u64, flops: u64, label: &str) -> Result<(), String> {
    if model != PerfModel::new(bytes, flops) {
        return Err(format!(
            "{label} accounting mismatch: got bytes={} flops={}, expected bytes={bytes} flops={flops}",
            model.bytes, model.flops
        ));
    }
    Ok(())
}

fn run() -> Result<Metrics, String> {
    let mut normal_cases = 0;
    expect(copy_h2d_model::<f64>(10), 80, 0, "copy_h2d")?;
    normal_cases += 1;
    expect(copy_d2d_model::<f32>(10), 80, 0, "copy_d2d")?;
    normal_cases += 1;
    expect(scale_model::<num_complex::Complex32>(10), 160, 60, "scale")?;
    normal_cases += 1;
    expect(dot_model::<num_complex::Complex32>(10), 160, 80, "dot")?;
    normal_cases += 1;
    expect(axpy_model::<num_complex::Complex32>(10), 240, 80, "axpy")?;
    normal_cases += 1;
    expect(nrm2_model::<num_complex::Complex32>(10), 80, 40, "nrm2")?;
    normal_cases += 1;
    expect(gemv_model::<f32>(4, 8), 192, 64, "gemv")?;
    normal_cases += 1;
    expect(gemm_model::<f64>(4, 8, 16), 2048, 1024, "gemm")?;
    normal_cases += 1;
    expect(
        cast_model::<f32, num_complex::Complex64>(10),
        200,
        0,
        "cast",
    )?;
    normal_cases += 1;
    expect(
        gather4_weighted_model::<num_complex::Complex32>(10),
        400,
        320,
        "gather4",
    )?;
    normal_cases += 1;
    // values 5*8 + col idx 5*4 + row ptr 4*4 + x 3*8 + y 3*8 = 124.
    // x traffic counts one element per column (perfect reuse), not per
    // nonzero; see csr_spmv_model_ncols and its unit test.
    expect(
        csr_spmv_model::<f64>(CsrRowBucketStats {
            rows: 3,
            nnz: 5,
            empty_rows: 0,
            short_rows: 3,
            simd_rows: 0,
            cta_rows: 0,
            max_nnz_per_row: 2,
        }),
        124,
        10,
        "csr_spmv",
    )?;
    normal_cases += 1;

    let extreme = usize::MAX;
    let saturation_models = [
        copy_h2d_model::<num_complex::Complex64>(extreme),
        copy_d2d_model::<num_complex::Complex64>(extreme),
        scale_model::<num_complex::Complex64>(extreme),
        dot_model::<num_complex::Complex64>(extreme),
        gemv_model::<num_complex::Complex64>(extreme, extreme),
        gemm_model::<num_complex::Complex64>(extreme, extreme, extreme),
        cast_model::<f32, num_complex::Complex64>(extreme),
        gather4_weighted_model::<num_complex::Complex64>(extreme),
        bem_regular_pair_c32_model(extreme),
    ];
    if saturation_models
        .iter()
        .any(|model| model.bytes != u64::MAX && model.flops != u64::MAX)
    {
        return Err("extreme roofline shape did not saturate accounting".into());
    }

    Ok(Metrics {
        case_count: normal_cases + saturation_models.len(),
        saturation_cases: saturation_models.len(),
        normal_cases,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("case_count", metrics.case_count as f64)
                .metric("normal_cases", metrics.normal_cases as f64)
                .metric("saturation_cases", metrics.saturation_cases as f64)
                .minimum("case_count", 20.0)
                .minimum("normal_cases", 11.0)
                .minimum("saturation_cases", 9.0)
                .reference("independent byte/FLOP formulas and saturating extreme-shape checks")
                .reference_checksum("analytic/deterministic-fixture:compute-perf-contract-v1")
                .dimension(
                    "CPU roofline models across BLAS, CSR, gather, cast, and BEM pair shapes",
                )
                .formulation("u64 saturating accounting")
                .tolerance("exact integer accounting")
                .seed("compute-perf-contract:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("performance contract result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("performance contract failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
