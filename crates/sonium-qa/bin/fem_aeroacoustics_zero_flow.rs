//! Aeroacoustics zero-flow benchmark
//!
//! Validates that the Linearized Potential Flow (LPF) assembler reduces
//! to the standard Helmholtz assembler when the mean-flow Mach number is
//! zero. At M=0 the convected Helmholtz equation must be identical to the
//! classical Helmholtz equation.
//!
//! Test: assemble both systems on the same mesh at the same frequency and
//! verify that every matrix entry agrees to within numerical tolerance.

use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::acoustics::aeroacoustics::{
    LpfAssemblerExt, MeanFlow, build_lpf_assembler,
};
use sonium_fem::physics::acoustics::pressure::{AcousticsAssemblerExt, build_acoustics_assembler};
use std::collections::HashMap;

fn test_zero_flow_matrix_equivalence() {
    println!("=== Aeroacoustics Zero-Flow Matrix Equivalence ===\n");

    let mesh = unit_square_triangles(8);
    let degree = PolynomialDegree::P1;
    let omega = 2000.0;

    let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]); // M = 0
    let lpf = build_lpf_assembler(&mesh, degree, &flow);
    let acoustics = build_acoustics_assembler(&mesh, degree);

    let k = Complex64::new(flow.wavenumber(omega), 0.0);
    let csr_lpf = lpf.assemble_convected_helmholtz(&flow, omega, &HashMap::new());
    let csr_helm = acoustics.assemble_helmholtz(k, &HashMap::new());

    assert_eq!(
        csr_lpf.num_rows, csr_helm.num_rows,
        "Matrix dimension mismatch"
    );
    assert_eq!(
        csr_lpf.num_cols, csr_helm.num_cols,
        "Matrix dimension mismatch"
    );

    let mut max_diff = 0.0_f64;
    let mut total_diff = 0.0_f64;
    let mut count = 0usize;

    for i in 0..csr_lpf.num_rows {
        let row_lpf: Vec<(usize, Complex64)> = csr_lpf.row_entries(i).collect();
        let row_helm: Vec<(usize, Complex64)> = csr_helm.row_entries(i).collect();

        assert_eq!(
            row_lpf.len(),
            row_helm.len(),
            "Row {i} nnz mismatch: lpf={} helm={}",
            row_lpf.len(),
            row_helm.len()
        );

        for ((j_lpf, v_lpf), (j_helm, v_helm)) in row_lpf.iter().zip(row_helm.iter()) {
            assert_eq!(
                j_lpf, j_helm,
                "Column index mismatch at row {i}: lpf={j_lpf} helm={j_helm}"
            );
            let diff = (v_lpf - v_helm).norm();
            max_diff = max_diff.max(diff);
            total_diff += diff;
            count += 1;
        }
    }

    let avg_diff = if count > 0 {
        total_diff / count as f64
    } else {
        0.0
    };

    println!("Matrix size: {} × {}", csr_lpf.num_rows, csr_lpf.num_cols);
    println!("Non-zeros compared: {count}");
    println!("Max entry difference: {:.6e}", max_diff);
    println!("Avg entry difference: {:.6e}", avg_diff);

    assert!(
        max_diff < 1e-10,
        "At M=0, LPF and Helmholtz matrices should be identical (max_diff={:.2e})",
        max_diff
    );

    println!("\n✓ PASSED: LPF reduces to Helmholtz at zero Mach number");
    println!("\n=== Benchmark Complete ===");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_zero_flow_matrix_equivalence();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
