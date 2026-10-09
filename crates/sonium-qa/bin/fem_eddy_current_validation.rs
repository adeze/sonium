//! Eddy current validation tests
//!
//! - Zero frequency: curl-curl matrix only (magnetostatic)
//! - Conductor vs air region behavior
//! - Frequency scaling

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::electromagnetics::eddy_current::{ConductorMaterial, EddyCurrentForm};
use sonium_fem::spaces::HcurlSpace;

fn test_eddy_current_zero_frequency() {
    let material = ConductorMaterial::copper();
    let form = EddyCurrentForm::new(&material, 0.0);

    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();

    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    let max_imag = csr.values.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
    assert!(max_imag < 1e-10, "Zero frequency should give real matrix");
}

fn test_eddy_current_form_matrices() {
    let material = ConductorMaterial::copper();
    let form = EddyCurrentForm::new(&material, 1000.0);

    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();

    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    assert!(csr.num_rows > 0, "Matrix should have rows");
    assert!(csr.nnz() > 0, "Matrix should be non-empty");

    let has_imag = csr.values.iter().any(|v| v.im.abs() > 1e-10);
    assert!(has_imag, "Should have imaginary part at non-zero frequency");
}

fn test_eddy_current_non_conducting() {
    let form = EddyCurrentForm::non_conducting(1.0);

    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();

    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    let max_imag = csr.values.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
    assert!(max_imag < 1e-10, "Non-conducting should give real matrix");
}

fn test_eddy_current_frequency_scaling() {
    let material = ConductorMaterial::copper();
    let f1: f64 = 100.0;
    let f2: f64 = 1000.0;

    let form1 = EddyCurrentForm::new(&material, f1);
    let form2 = EddyCurrentForm::new(&material, f2);

    let omega1 = 2.0 * std::f64::consts::PI * f1;
    let omega2 = 2.0 * std::f64::consts::PI * f2;

    assert!((form2.omega_sigma / form1.omega_sigma - omega2 / omega1).abs() < 1e-10);
}

fn test_eddy_current_complex_symmetry() {
    use num_complex::Complex64;
    let material = ConductorMaterial::aluminum();
    let form = EddyCurrentForm::new(&material, 500.0);

    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();

    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    let n = csr.num_rows.min(10);
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
            let j = csr.col_indices[k];
            if j < n {
                let aij = csr.values[k];
                let aji = (csr.row_ptrs[j]..csr.row_ptrs[j + 1])
                    .find(|&kk| csr.col_indices[kk] == i)
                    .map(|kk| csr.values[kk])
                    .unwrap_or(Complex64::new(0.0, 0.0));
                let diff = (aij - aji).norm();
                max_asym = max_asym.max(diff);
            }
        }
    }
    assert!(
        max_asym < 1e-10,
        "Complex symmetric eddy-current matrix violated: max |A[i,j] - A[j,i]| = {max_asym:.2e}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_eddy_current_zero_frequency();
        test_eddy_current_form_matrices();
        test_eddy_current_non_conducting();
        test_eddy_current_frequency_scaling();
        test_eddy_current_complex_symmetry();
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
