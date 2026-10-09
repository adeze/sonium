//! FEM1 bounded native CUDA parity for independent P1/Tri3 local tensors.
//!
//! The local element oracle is assembled analytically on the host and packed
//! as two independent blocks. CUDA executes the supported resident CSR action
//! over those blocks; this packet deliberately does not advertise a CUDA
//! element or quadrature kernel.

use num_complex::Complex64;
use serde_json::json;
use sha2::{Digest, Sha256};
use sonium_compute::{
    AdjointDeviceCsr, Backend, DeviceAdjointOperator, DeviceOperator, HostCsrRef,
};
use sonium_qa::{cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};

const SCHEMA: &str = "sonium.fem.element-local-cuda-parity.v1";
const CASE_ID: &str = "fem1.element-local-cuda.v1";
const ELEMENTS: usize = 2;
const LOCAL_DOFS: usize = 3;
const DOFS: usize = ELEMENTS * LOCAL_DOFS;
const RESIDUAL_TOLERANCE: f64 = 2.0e-12;
const INNER_PRODUCT_TOLERANCE: f64 = 2.0e-12;

fn sha256_hex(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn local_tensor() -> [Complex64; 9] {
    let gradients = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
    let area = 0.5;
    let wavenumber_squared = Complex64::new(2.25, 0.07);
    let mut tensor = [Complex64::default(); 9];
    for row in 0..LOCAL_DOFS {
        for col in 0..LOCAL_DOFS {
            let stiffness = area
                * gradients[row]
                    .iter()
                    .zip(gradients[col])
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
            let mass = area / 12.0 * if row == col { 2.0 } else { 1.0 };
            tensor[row * LOCAL_DOFS + col] =
                Complex64::new(stiffness, 0.0) - wavenumber_squared * mass;
        }
    }
    tensor
}

fn block_diagonal_csr() -> (Vec<usize>, Vec<usize>, Vec<Complex64>) {
    let local = local_tensor();
    let mut row_ptrs = Vec::with_capacity(DOFS + 1);
    let mut cols = Vec::with_capacity(ELEMENTS * LOCAL_DOFS * LOCAL_DOFS);
    let mut values = Vec::with_capacity(ELEMENTS * LOCAL_DOFS * LOCAL_DOFS);
    row_ptrs.push(0);
    for element in 0..ELEMENTS {
        let offset = element * LOCAL_DOFS;
        for row in 0..LOCAL_DOFS {
            for col in 0..LOCAL_DOFS {
                cols.push(offset + col);
                values.push(local[row * LOCAL_DOFS + col]);
            }
            row_ptrs.push(cols.len());
        }
    }
    (row_ptrs, cols, values)
}

fn apply_reference(
    row_ptrs: &[usize],
    cols: &[usize],
    values: &[Complex64],
    input: &[Complex64],
    hermitian: bool,
) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); DOFS];
    for row in 0..DOFS {
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            let value = if hermitian {
                values[index].conj()
            } else {
                values[index]
            };
            if hermitian {
                output[cols[index]] += value * input[row];
            } else {
                output[row] += value * input[cols[index]];
            }
        }
    }
    output
}

fn max_relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm() / expected.norm().max(1.0e-30))
        .fold(0.0, f64::max)
}

fn dotc(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

fn main() {
    assert_eq!(std::env::consts::OS, "linux", "FEM1 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "FEM1 requires native x86_64"
    );

    let backend = required_cuda_backend().expect("required SM120 CUDA backend");
    backend.reset_runtime_stats();
    let (row_ptrs, cols, values) = block_diagonal_csr();
    let host = HostCsrRef::try_new(DOFS, DOFS, &row_ptrs, &cols, &values)
        .expect("independent local block CSR admission");
    let operator = AdjointDeviceCsr::<_, Complex64>::try_from_host_csr_ref(&backend, host)
        .expect("upload local block CSR to CUDA");

    let input: Vec<_> = (0..DOFS)
        .map(|index| Complex64::new((index as f64 + 0.25).sin(), (index as f64 * 0.31).cos()))
        .collect();
    let test_vector: Vec<_> = (0..DOFS)
        .map(|index| Complex64::new((index as f64 * 0.17).cos(), (index as f64 + 0.5).sin()))
        .collect();
    let mut device_input = backend
        .raw_alloc::<Complex64>(DOFS)
        .expect("input allocation");
    let mut device_test = backend
        .raw_alloc::<Complex64>(DOFS)
        .expect("test allocation");
    backend
        .try_copy_h2d(&input, &mut device_input)
        .expect("input upload");
    backend
        .try_copy_h2d(&test_vector, &mut device_test)
        .expect("test upload");
    let mut device_forward = backend
        .raw_alloc::<Complex64>(DOFS)
        .expect("forward allocation");
    let mut device_adjoint = backend
        .raw_alloc::<Complex64>(DOFS)
        .expect("adjoint allocation");
    operator
        .try_apply(&backend, &device_input, &mut device_forward)
        .expect("CUDA forward local block action");
    operator
        .try_apply_hermitian(&backend, &device_test, &mut device_adjoint)
        .expect("CUDA Hermitian local block action");
    let mut forward = vec![Complex64::default(); DOFS];
    let mut adjoint = vec![Complex64::default(); DOFS];
    backend
        .try_copy_d2h(&device_forward, &mut forward)
        .expect("forward download");
    backend
        .try_copy_d2h(&device_adjoint, &mut adjoint)
        .expect("adjoint download");

    let forward_reference = apply_reference(&row_ptrs, &cols, &values, &input, false);
    let adjoint_reference = apply_reference(&row_ptrs, &cols, &values, &test_vector, true);
    let forward_error = max_relative_error(&forward, &forward_reference);
    let adjoint_error = max_relative_error(&adjoint, &adjoint_reference);
    let lhs = dotc(&forward, &test_vector);
    let rhs = dotc(&input, &adjoint);
    let inner_product_error = (lhs - rhs).norm() / lhs.norm().max(1.0e-30);

    let mut isolated_input = vec![Complex64::default(); DOFS];
    isolated_input[..LOCAL_DOFS].copy_from_slice(&input[..LOCAL_DOFS]);
    let mut isolated_host = backend
        .raw_alloc::<Complex64>(DOFS)
        .expect("isolation input allocation");
    let mut isolated_device = backend
        .raw_alloc::<Complex64>(DOFS)
        .expect("isolation output allocation");
    backend
        .try_copy_h2d(&isolated_input, &mut isolated_host)
        .expect("isolation input upload");
    operator
        .try_apply(&backend, &isolated_host, &mut isolated_device)
        .expect("CUDA isolated local block action");
    let mut isolated = vec![Complex64::default(); DOFS];
    backend
        .try_copy_d2h(&isolated_device, &mut isolated)
        .expect("isolation download");
    let isolated_reference = apply_reference(&row_ptrs, &cols, &values, &isolated_input, false);
    let isolation_error = max_relative_error(&isolated, &isolated_reference);
    let second_block_leakage = isolated[LOCAL_DOFS..]
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);

    assert!(forward_error <= RESIDUAL_TOLERANCE);
    assert!(adjoint_error <= RESIDUAL_TOLERANCE);
    assert!(inner_product_error <= INNER_PRODUCT_TOLERANCE);
    assert!(isolation_error <= RESIDUAL_TOLERANCE);
    assert!(second_block_leakage <= RESIDUAL_TOLERANCE);

    let bad_input = backend
        .raw_alloc::<Complex64>(DOFS - 1)
        .expect("bad input allocation");
    let mut bad_output = backend
        .raw_alloc::<Complex64>(DOFS)
        .expect("bad output allocation");
    let typed_shape_error = operator
        .try_apply(&backend, &bad_input, &mut bad_output)
        .expect_err("wrong-sized local action must be rejected");
    let typed_shape_error = format!("{typed_shape_error:?}");
    assert!(typed_shape_error.contains("InvalidArgument"));

    let canonical_input = format!(
        "schema={SCHEMA};case_id={CASE_ID};elements={ELEMENTS};local_dofs={LOCAL_DOFS};area_m2=0.5;gradients=[[-1,-1],[1,0],[0,1]];wavenumber_squared=2.25+0.07i;representation=block_diagonal_csr"
    );
    let stats = backend.runtime_stats();
    let payload = json!({
        "schema": SCHEMA,
        "status": "passed",
        "case_id": CASE_ID,
        "scope": "native Linux x86_64 required CUDA SM120; Windows/Arm64/AVX512 waived",
        "claim_class": "independent P1/Tri3 local Helmholtz tensor parity through resident CUDA CSR action",
        "reference": "area=0.5, gradients=[[-1,-1],[1,0],[0,1]], consistent mass area/12, and k²=2.25+0.07i",
        "canonical_input": canonical_input,
        "canonical_input_sha256": sha256_hex(&canonical_input),
        "elements": ELEMENTS,
        "local_dofs": LOCAL_DOFS,
        "dofs": DOFS,
        "nnz": values.len(),
        "metrics": {
            "forward_relative_error": forward_error,
            "hermitian_relative_error": adjoint_error,
            "inner_product_error": inner_product_error,
            "isolated_block_relative_error": isolation_error,
            "second_block_leakage": second_block_leakage,
            "local_tensor_entries": values.len() / ELEMENTS,
        },
        "thresholds": {
            "forward_relative_error": RESIDUAL_TOLERANCE,
            "hermitian_relative_error": RESIDUAL_TOLERANCE,
            "inner_product_error": INNER_PRODUCT_TOLERANCE,
            "isolated_block_relative_error": RESIDUAL_TOLERANCE,
            "second_block_leakage": RESIDUAL_TOLERANCE,
        },
        "runtime": {
            "kernel_launches": stats.kernel_launches,
            "library_operations": stats.library_operations,
            "synchronization_calls": stats.synchronization_calls,
            "failed_synchronizations": stats.failed_synchronizations,
            "allocation_operations": stats.allocation_operations,
            "staging_allocation_operations": stats.staging_allocation_operations,
            "h2d_operations": stats.transfer.h2d_operations,
            "h2d_bytes": stats.transfer.h2d_bytes,
            "d2h_operations": stats.transfer.d2h_operations,
            "d2h_bytes": stats.transfer.d2h_bytes,
            "d2d_operations": stats.transfer.d2d_operations,
            "d2d_bytes": stats.transfer.d2d_bytes,
            "live_allocation_bytes": stats.live_allocation_bytes,
            "allocation_high_water_bytes": stats.allocation_high_water_bytes,
            "live_staging_bytes": stats.live_staging_bytes,
            "staging_high_water_bytes": stats.staging_high_water_bytes,
        },
        "admission": {
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "dispatch": true,
            "resident_structure": "device CSR descriptor and vectors",
            "unsupported_element_kernel": "explicitly not claimed",
        },
        "failure_contracts": [{
            "id": "wrong_vector_shape",
            "status": "typed_rejected",
            "stage": "action_admission",
            "error": typed_shape_error,
            "dispatch": false,
        }],
        "unsupported_scope": [
            "native CUDA element/quadrature tensor kernel remains unsupported",
            "this packet does not claim partial assembly or general matrix-free FEM execution",
        ],
        "hardware": cuda_hardware_provenance(),
    });
    let bytes =
        serde_json::to_vec_pretty(&payload).expect("serialize FEM1 element parity artifact");
    let (path, checksum) = write_qa_artifact("fem1_element_local_cuda.json", &bytes)
        .unwrap_or_else(|error| {
            eprintln!("FAILED: write FEM1 element parity artifact: {error}");
            std::process::exit(1)
        });
    println!("artifact={path} checksum={checksum}");
    println!("SUCCESS");
}
