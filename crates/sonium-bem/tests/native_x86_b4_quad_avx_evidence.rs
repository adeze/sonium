//! Native x86 B4 evidence for the production separated Quad4 AVX2/FMA cell.
//!
//! The scalar implementation remains the oracle.  This packet compares the
//! runtime-dispatched production path with that oracle for linear and
//! quadratic Quad4 bases over low/acoustic/high phase regimes and both
//! SIMD-aligned and scalar-tail quadrature orders.  Singular/touching charts,
//! weak-form derivatives, Arm64, Windows, and AVX512 are outside this cell.

#![cfg(target_arch = "x86_64")]

use num_complex::Complex64;
use serde::Serialize;
use sha2::{Digest, Sha256};
use sonium_bem::core::integration::duffy::{
    GalerkinIntegrationResult, galerkin_single_layer_regular_quad4_dispatch,
    galerkin_single_layer_regular_quad4_with_basis,
    galerkin_single_layer_regular_quad4_with_basis_scalar,
};
use sonium_bem::core::types::PhysicsParams;
use sonium_solvers::dense::DenseMatrix;
use std::process::Command;

const SCHEMA: &str = "sonium-bem-native-x86-b4-quad-avx/v1";
const PHASES: [(&str, f64); 3] = [("low", 42.0), ("acoustic", 420.0), ("high", 1_000.0)];
const BASES: [(&str, usize); 2] = [("linear", 4), ("quadratic", 9)];
const GAUSS_ORDERS: [usize; 3] = [1, 2, 4];
const CASE_COUNT: usize = PHASES.len() * BASES.len() * GAUSS_ORDERS.len();
const PARITY_BUDGET: f64 = 5.0e-11;

#[derive(Debug, Serialize)]
struct CaseReport {
    phase: &'static str,
    frequency_hz: f64,
    basis: &'static str,
    basis_count: usize,
    gauss_order: usize,
    effective_order: usize,
    field_quadrature_points: usize,
    avx2_batches: usize,
    scalar_tail_points: usize,
    complex_observables: usize,
    max_abs_error: f64,
    max_relative_error: f64,
}

#[derive(Debug, Serialize)]
struct AvxCapability {
    avx2: bool,
    fma: bool,
    avx512f: bool,
    selected_kernel: &'static str,
    scalar_oracle: &'static str,
    precision: &'static str,
}

#[derive(Debug, Serialize)]
struct AvxArtifact {
    schema: &'static str,
    status: &'static str,
    target_arch: &'static str,
    os: &'static str,
    cpu_model: String,
    microcode: String,
    compiler: String,
    capability: AvxCapability,
    element_family: &'static str,
    pair_kind: &'static str,
    phase_regimes: &'static str,
    basis_orders: &'static str,
    gauss_orders: &'static str,
    tail_policy: &'static str,
    parity_budget: f64,
    cases_expected: usize,
    cases_passed: usize,
    cases_failed: usize,
    max_abs_error: f64,
    max_relative_error: f64,
    cases: Vec<CaseReport>,
    evidence_sha256: String,
    unsupported_scope: &'static str,
}

fn cpuinfo_value(field: &str) -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == field).then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unknown".into())
}

fn compiler_version() -> String {
    Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| output.lines().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".into())
}

fn square(z: f64) -> DenseMatrix<f64> {
    DenseMatrix::from_row_major_vec(
        4,
        3,
        vec![-0.5, -0.5, z, 0.5, -0.5, z, 0.5, 0.5, z, -0.5, 0.5, z],
    )
}

fn complex_values(result: &GalerkinIntegrationResult) -> Vec<Complex64> {
    let mut values = Vec::new();
    for row in &result.basis_results {
        for entry in row {
            values.extend([
                entry.g_integral,
                entry.dg_dn_integral,
                entry.dg_dnx_integral,
                entry.d2g_dnxdny_integral,
            ]);
        }
    }
    for row in &result.curl_integrals {
        values.extend(row.iter().copied());
    }
    for row in &result.normal_mass_integrals {
        values.extend(row.iter().copied());
    }
    values
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let absolute = (*actual - *expected).norm();
            let relative = absolute / expected.norm().max(1.0e-30);
            (absolute, relative)
        })
        .fold(
            (0.0, 0.0),
            |(max_abs, max_relative), (absolute, relative)| {
                (max_abs.max(absolute), max_relative.max(relative))
            },
        )
}

fn digest_cases(cases: &[CaseReport]) -> String {
    let bytes = serde_json::to_vec(cases).expect("serialize AVX Quad4 cases");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn native_x86_b4_quad4_avx2_fma_parity_v1() {
    assert_eq!(std::env::consts::OS, "linux");
    assert!(std::is_x86_feature_detected!("avx2"), "AVX2 is required");
    assert!(std::is_x86_feature_detected!("fma"), "FMA is required");
    let selected_kernel = galerkin_single_layer_regular_quad4_dispatch();
    assert_eq!(selected_kernel, "quad4-regular-avx2-fma");

    let source = square(0.0);
    let field = square(1.25);
    let mut cases = Vec::with_capacity(CASE_COUNT);
    let mut max_abs_error = 0.0_f64;
    let mut max_relative_error = 0.0_f64;

    for &(phase, frequency_hz) in &PHASES {
        let physics = PhysicsParams::try_new(frequency_hz, 343.0, 1.21, false)
            .expect("valid Quad4 phase regime");
        for &(basis, basis_count) in &BASES {
            for &gauss_order in &GAUSS_ORDERS {
                let actual = galerkin_single_layer_regular_quad4_with_basis(
                    &source,
                    &field,
                    &physics,
                    basis_count,
                    basis_count,
                    gauss_order,
                );
                let expected = galerkin_single_layer_regular_quad4_with_basis_scalar(
                    &source,
                    &field,
                    &physics,
                    basis_count,
                    basis_count,
                    gauss_order,
                );
                let actual = complex_values(&actual);
                let expected = complex_values(&expected);
                let (case_abs, case_relative) = error_metrics(&actual, &expected);
                assert!(
                    case_abs <= PARITY_BUDGET && case_relative <= PARITY_BUDGET,
                    "AVX Quad4 parity failed phase={phase}, basis={basis}, order={gauss_order}: abs={case_abs:.3e}, rel={case_relative:.3e}"
                );
                let effective_order = gauss_order + 7;
                let field_points = effective_order * effective_order;
                let avx2_batches = effective_order * effective_order * (field_points / 4);
                let scalar_tail_points = effective_order * effective_order * (field_points % 4);
                cases.push(CaseReport {
                    phase,
                    frequency_hz,
                    basis,
                    basis_count,
                    gauss_order,
                    effective_order,
                    field_quadrature_points: field_points,
                    avx2_batches,
                    scalar_tail_points,
                    complex_observables: actual.len(),
                    max_abs_error: case_abs,
                    max_relative_error: case_relative,
                });
                max_abs_error = max_abs_error.max(case_abs);
                max_relative_error = max_relative_error.max(case_relative);
            }
        }
    }
    assert_eq!(cases.len(), CASE_COUNT);
    assert!(cases.iter().any(|case| case.scalar_tail_points > 0));
    assert!(cases.iter().any(|case| case.avx2_batches > 0));

    let artifact = AvxArtifact {
        schema: SCHEMA,
        status: "pass",
        target_arch: "x86_64",
        os: "linux-x86_64",
        cpu_model: cpuinfo_value("model name"),
        microcode: cpuinfo_value("microcode"),
        compiler: compiler_version(),
        capability: AvxCapability {
            avx2: true,
            fma: true,
            avx512f: std::is_x86_feature_detected!("avx512f"),
            selected_kernel,
            scalar_oracle: "galerkin_single_layer_regular_quad4_with_basis_scalar",
            precision: "Complex64",
        },
        element_family: "Quad4",
        pair_kind: "separated-regular",
        phase_regimes: "low:42Hz,acoustic:420Hz,high:1000Hz",
        basis_orders: "linear:4,quadratic:9",
        gauss_orders: "1,2,4 (effective 8,9,11)",
        tail_policy: "four field quadrature points per AVX2/FMA batch; scalar remainder",
        parity_budget: PARITY_BUDGET,
        cases_expected: CASE_COUNT,
        cases_passed: cases.len(),
        cases_failed: 0,
        max_abs_error,
        max_relative_error,
        evidence_sha256: digest_cases(&cases),
        cases,
        unsupported_scope: "no Quad4 singular/touching AVX claim, weak-form derivative AVX claim, f32/C32 Quad4 claim, AVX512, Arm64, or Windows claim",
    };
    eprintln!(
        "sonium_bem_native_x86_b4_quad_avx={{\"schema\":\"{}\",\"status\":\"{}\",\"selected_kernel\":\"{}\",\"cases_passed\":{},\"max_abs_error\":{:.3e},\"max_relative_error\":{:.3e}}}",
        artifact.schema,
        artifact.status,
        artifact.capability.selected_kernel,
        artifact.cases_passed,
        artifact.max_abs_error,
        artifact.max_relative_error,
    );
    if let Some(output_dir) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let output_dir = std::path::PathBuf::from(output_dir);
        std::fs::create_dir_all(&output_dir).expect("create AVX Quad4 evidence directory");
        let json = serde_json::to_vec_pretty(&artifact).expect("serialize AVX Quad4 artifact");
        std::fs::write(
            output_dir.join("sonium_bem_native_x86_b4_quad_avx.json"),
            json,
        )
        .expect("write AVX Quad4 artifact");
    }
}
