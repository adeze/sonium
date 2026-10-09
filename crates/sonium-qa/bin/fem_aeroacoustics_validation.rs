//! Aeroacoustics validation tests
//!
//! Tests for Linearized Potential Flow (LPF) forms.
//!
//! - Zero flow limit: convection should give zero matrix
//! - Streaming stiffness symmetry
//! - Convection skew-symmetry

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::physics::acoustics::aeroacoustics::{ScalarConvectionForm, StreamingStiffnessForm};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn test_lpf_zero_mach_reduces_to_helmholtz() {
    let mesh = sonium_fem::mesh::generators::unit_square_triangles(8);
    let space = H1Space::new(PolynomialDegree::P1);

    let conv_form = ScalarConvectionForm {
        advection: [0.0, 0.0, 0.0],
    };
    let triplet = assemble_bilinear(&conv_form, &space, &space, &mesh, 1);
    let csr = triplet.to_csr();

    let nnz = csr.nnz();
    let max_val: f64 = csr.values.iter().map(|v| v.abs()).fold(0.0, f64::max);

    assert!(
        nnz == 0 || max_val < 1e-10,
        "Zero Mach convection should give zero matrix"
    );
}

fn test_lpf_streaming_stiffness_symmetric() {
    let mach = [0.3, 0.0, 0.0];
    let mesh = sonium_fem::mesh::generators::unit_square_triangles(4);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = StreamingStiffnessForm { mach };
    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let csr = triplet.to_csr();

    let n = csr.num_rows;
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
            let j = csr.col_indices[k];
            let aij = csr.values[k];
            let aji = (csr.row_ptrs[j]..csr.row_ptrs[j + 1])
                .find(|&kk| csr.col_indices[kk] == i)
                .map(|kk| csr.values[kk])
                .unwrap_or(0.0);
            max_asym = max_asym.max((aij - aji).abs());
        }
    }

    assert!(
        max_asym < 1e-10,
        "Streaming stiffness not symmetric: {:.2e}",
        max_asym
    );
}

fn test_lpf_convection_skew_symmetric() {
    let mach = [0.2, 0.1, 0.0];
    let mesh = sonium_fem::mesh::generators::unit_square_triangles(4);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = ScalarConvectionForm { advection: mach };
    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let csr = triplet.to_csr();

    let n = csr.num_rows;
    let mut total_skew = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
            let j = csr.col_indices[k];
            if j > i {
                let aij = csr.values[k];
                let aji = (csr.row_ptrs[j]..csr.row_ptrs[j + 1])
                    .find(|&kk| csr.col_indices[kk] == i)
                    .map(|kk| csr.values[kk])
                    .unwrap_or(0.0);
                total_skew += (aij + aji).abs();
                count += 1;
            }
        }
    }

    let avg_skew = if count > 0 {
        total_skew / count as f64
    } else {
        0.0
    };

    assert!(
        avg_skew < 1e-10,
        "Convection should be skew-symmetric (A + A^T ≈ 0): {:.2e}",
        avg_skew
    );
}

fn test_streaming_stiffness_scaling() {
    let mesh = sonium_fem::mesh::generators::unit_square_triangles(4);
    let space = H1Space::new(PolynomialDegree::P1);

    let mach1 = [0.1, 0.0, 0.0];
    let mach2 = [0.2, 0.0, 0.0];

    let form1 = StreamingStiffnessForm { mach: mach1 };
    let form2 = StreamingStiffnessForm { mach: mach2 };

    let triplet1 = assemble_bilinear(&form1, &space, &space, &mesh, 1);
    let triplet2 = assemble_bilinear(&form2, &space, &space, &mesh, 1);

    let norm1: f64 = triplet1.values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm2: f64 = triplet2.values.iter().map(|v| v * v).sum::<f64>().sqrt();

    assert!(norm2 > norm1, "Larger Mach should give larger matrix norm");
}

fn test_convection_form_assembly() {
    let advection_2d = [0.5, 0.3, 0.0];
    let mesh = sonium_fem::mesh::generators::unit_square_triangles(2);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = ScalarConvectionForm {
        advection: advection_2d,
    };
    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let n_dofs = space.num_global_dofs(&mesh);
    assert!(triplet.num_rows == n_dofs);
    assert!(triplet.num_cols == n_dofs);
    assert!(triplet.nnz() > 0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_lpf_zero_mach_reduces_to_helmholtz();
        test_lpf_streaming_stiffness_symmetric();
        test_lpf_convection_skew_symmetric();
        test_streaming_stiffness_scaling();
        test_convection_form_assembly();
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
