//! CPU parity contract for the complex 4M and native GEMV implementations.

use num_complex::{Complex32, Complex64};
use sonium_compute::{
    Transpose,
    cpu::blas2::{complex_native, complex4m},
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.complex-gemv-parity.v1";
const C32_TOLERANCE: f64 = 2.0e-5;
const C64_TOLERANCE: f64 = 2.0e-12;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    complex32_error: f64,
    complex64_error: f64,
    complex32_cases: usize,
    complex64_cases: usize,
}

fn relative_error32(actual: Complex32, expected: Complex32) -> f64 {
    f64::from((actual - expected).norm()) / f64::from(expected.norm().max(1.0))
}

fn relative_error64(actual: Complex64, expected: Complex64) -> f64 {
    f64::from((actual - expected).norm()) / expected.norm().max(1.0)
}

fn run() -> Result<Metrics, String> {
    let mut maximum32 = 0.0_f64;
    let mut maximum64 = 0.0_f64;
    let mut cases32 = 0;
    let mut cases64 = 0;

    for (case, &(m, n)) in [(1, 1), (5, 3), (7, 5), (11, 4)].iter().enumerate() {
        // Allocate enough stored rows for both the normal and transposed
        // interpretations; the API always returns m outputs from n inputs.
        let lda = m.max(n) + 1;
        let stored_rows = m.max(n);
        let matrix32: Vec<Complex32> = (0..stored_rows * lda)
            .map(|index| {
                let row = index / lda;
                let column = index % lda;
                if column < n && row < m {
                    Complex32::new(
                        0.17 * (row as f32 + 1.0) - 0.09 * column as f32 + case as f32 * 0.03,
                        -0.21 + 0.07 * (row + column) as f32,
                    )
                } else {
                    Complex32::new(37.0, -19.0)
                }
            })
            .collect();
        let matrix64: Vec<Complex64> = matrix32
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect();
        let x32: Vec<Complex32> = (0..n)
            .map(|index| Complex32::new(0.23 + index as f32 * 0.11, -0.14 + index as f32 * 0.08))
            .collect();
        let x64: Vec<Complex64> = x32
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect();
        let initial32: Vec<Complex32> = (0..m)
            .map(|index| Complex32::new(-0.31 + index as f32 * 0.05, 0.18 - index as f32 * 0.04))
            .collect();
        let initial64: Vec<Complex64> = initial32
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect();
        let alpha32 = Complex32::new(0.63 + case as f32 * 0.02, -0.37);
        let beta32 = Complex32::new(-0.22, 0.19 + case as f32 * 0.01);
        let alpha64 = Complex64::new(f64::from(alpha32.re), f64::from(alpha32.im));
        let beta64 = Complex64::new(f64::from(beta32.re), f64::from(beta32.im));

        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            let mut four_m32 = initial32.clone();
            let mut native32 = initial32.clone();
            complex4m::gemv_c32(
                op,
                m,
                n,
                alpha32,
                &matrix32,
                lda,
                &x32,
                beta32,
                &mut four_m32,
            );
            complex_native::gemv_c32(
                op,
                m,
                n,
                alpha32,
                &matrix32,
                lda,
                &x32,
                beta32,
                &mut native32,
            );
            for (&actual, &expected) in four_m32.iter().zip(&native32) {
                maximum32 = maximum32.max(relative_error32(actual, expected));
            }
            cases32 += 1;

            let mut four_m64 = initial64.clone();
            let mut native64 = initial64.clone();
            complex4m::gemv_c64(
                op,
                m,
                n,
                alpha64,
                &matrix64,
                lda,
                &x64,
                beta64,
                &mut four_m64,
            );
            complex_native::gemv_c64(
                op,
                m,
                n,
                alpha64,
                &matrix64,
                lda,
                &x64,
                beta64,
                &mut native64,
            );
            for (&actual, &expected) in four_m64.iter().zip(&native64) {
                maximum64 = maximum64.max(relative_error64(actual, expected));
            }
            cases64 += 1;
        }
    }

    if maximum32 > C32_TOLERANCE || maximum64 > C64_TOLERANCE {
        return Err(format!(
            "complex GEMV implementation parity exceeded tolerance: f32={maximum32:.3e}, f64={maximum64:.3e}"
        ));
    }
    Ok(Metrics {
        complex32_error: maximum32,
        complex64_error: maximum64,
        complex32_cases: cases32,
        complex64_cases: cases64,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("complex4m and complex_native direct implementation parity")
                .reference_checksum("analytic/deterministic-fixture:compute-complex-gemv-parity-v1")
                .dimension("4 rectangular fixtures; padded row-major lda; NoTrans/Trans/ConjTrans")
                .formulation("CPU Complex32/Complex64 GEMV implementation comparison")
                .metric("complex32_max_relative_error", metrics.complex32_error)
                .metric("complex64_max_relative_error", metrics.complex64_error)
                .metric("complex32_case_count", metrics.complex32_cases as f64)
                .metric("complex64_case_count", metrics.complex64_cases as f64)
                .maximum("complex32_max_relative_error", C32_TOLERANCE)
                .maximum("complex64_max_relative_error", C64_TOLERANCE)
                .minimum("complex32_case_count", 12.0)
                .minimum("complex64_case_count", 12.0)
                .seed("compute-complex-gemv-parity:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("complex GEMV parity QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("complex GEMV parity QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
