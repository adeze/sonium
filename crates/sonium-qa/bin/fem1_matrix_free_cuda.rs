//! FEM1 bounded native CUDA matrix-free action for production P1/Tri3 cells.
//!
//! The production Helmholtz form and generic FEM assembly layer prepare one
//! local tensor per disconnected element on the host.  CUDA never receives a
//! global CSR matrix: it applies those resident tensors to resident local
//! vectors through `Backend::try_batched_local_contract`.  The independent
//! affine-triangle tensor and action reference keeps the check separate from
//! both the form and the CUDA implementation.

use std::time::Instant;

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendError, CudaBackend, CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot,
};
use sonium_fem::assembly::assemble_bilinear_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::acoustics::pressure::HelmholtzForm;
use sonium_fem::spaces::H1Space;
use sonium_fem::{
    FemAction, FemActualBackend, FemBackend, FemBackendRequest, FemScalar, FemWorkload,
    admit_fem_workload, fem_cuda_capability_rows,
};
use sonium_qa::{cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};

const SCHEMA: &str = "sonium-compute-cuda-fem-matrix-free/v1";
const CASE_ID: &str = "compute.fem-matrix-free-cuda.v1";
const ELEMENTS: usize = 3;
const LOCAL_DOFS: usize = 3;
const DOFS: usize = ELEMENTS * LOCAL_DOFS;
const WAVENUMBER: Complex64 = Complex64::new(1.7, 0.13);
const ACTION_TOLERANCE: f64 = 2.0e-12;
const TAYLOR_TOLERANCE: f64 = 2.0e-12;

#[derive(Debug)]
struct Metrics {
    production_reference_error: f64,
    forward_error: f64,
    transpose_error: f64,
    hermitian_error: f64,
    inner_product_error: f64,
    taylor_error: f64,
    warm_forward_error: f64,
    warm_transpose_error: f64,
    warm_hermitian_error: f64,
    retry_error: f64,
    warm_kernel_launches: usize,
    warm_h2d_operations: usize,
    warm_allocation_operations: usize,
    warm_staging_allocation_operations: usize,
    failure_typed: bool,
    output_preserved: bool,
    retry_succeeded: bool,
    malformed_rejected: bool,
    malformed_output_preserved: bool,
    empty_batch_accepted: bool,
    elapsed_ms: u64,
}

fn build_fixture() -> (Mesh, [[Point; LOCAL_DOFS]; ELEMENTS]) {
    let points = [
        [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
        ],
        [
            Point::new_2d(2.0, 0.0),
            Point::new_2d(3.0, 0.0),
            Point::new_2d(2.0, 2.0),
        ],
        [
            Point::new_2d(0.0, 3.0),
            Point::new_2d(1.0, 3.0),
            Point::new_2d(0.0, 4.0),
        ],
    ];
    let mut mesh = Mesh::new(2);
    for element in points {
        for point in element {
            mesh.add_node(point);
        }
    }
    for element in 0..ELEMENTS {
        let offset = element * LOCAL_DOFS;
        mesh.add_element(ElementType::Triangle, vec![offset, offset + 1, offset + 2]);
    }
    (mesh, points)
}

/// Independent affine P1/Tri3 reference, not using the production tabulator.
fn independent_tensor(points: [Point; LOCAL_DOFS]) -> [Complex64; LOCAL_DOFS * LOCAL_DOFS] {
    let twice_signed_area = (points[1].x - points[0].x) * (points[2].y - points[0].y)
        - (points[2].x - points[0].x) * (points[1].y - points[0].y);
    let area = 0.5 * twice_signed_area.abs();
    assert!(
        twice_signed_area > 0.0,
        "fixture triangles must be positively oriented"
    );
    let gradients = [
        [
            (points[1].y - points[2].y) / twice_signed_area,
            (points[2].x - points[1].x) / twice_signed_area,
        ],
        [
            (points[2].y - points[0].y) / twice_signed_area,
            (points[0].x - points[2].x) / twice_signed_area,
        ],
        [
            (points[0].y - points[1].y) / twice_signed_area,
            (points[1].x - points[0].x) / twice_signed_area,
        ],
    ];
    let k_squared = WAVENUMBER * WAVENUMBER;
    let mut tensor = [Complex64::default(); LOCAL_DOFS * LOCAL_DOFS];
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
                Complex64::new(stiffness, 0.0) - k_squared * Complex64::new(mass, 0.0);
        }
    }
    tensor
}

fn production_tensors(
    assembled: &sonium_fem::assembly::TripletMatrix<Complex64>,
) -> [[[Complex64; LOCAL_DOFS * LOCAL_DOFS]; ELEMENTS]; 1] {
    let mut tensors = [[[Complex64::default(); LOCAL_DOFS * LOCAL_DOFS]; ELEMENTS]; 1];
    for element in 0..ELEMENTS {
        let offset = element * LOCAL_DOFS;
        for row in 0..LOCAL_DOFS {
            for col in 0..LOCAL_DOFS {
                let global_row = offset + row;
                let global_col = offset + col;
                tensors[0][element][row * LOCAL_DOFS + col] = assembled
                    .rows
                    .iter()
                    .zip(&assembled.cols)
                    .zip(&assembled.values)
                    .filter(|((stored_row, stored_col), _)| {
                        **stored_row == global_row && **stored_col == global_col
                    })
                    .map(|((_, _), value)| *value)
                    .sum();
            }
        }
    }
    tensors
}

fn flatten_tensors(tensors: &[[Complex64; LOCAL_DOFS * LOCAL_DOFS]; ELEMENTS]) -> Vec<Complex64> {
    tensors
        .iter()
        .flat_map(|tensor| tensor.iter().copied())
        .collect()
}

fn transpose_tensors(
    tensors: &[[Complex64; LOCAL_DOFS * LOCAL_DOFS]; ELEMENTS],
    conjugate: bool,
) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); DOFS * LOCAL_DOFS];
    for element in 0..ELEMENTS {
        for row in 0..LOCAL_DOFS {
            for col in 0..LOCAL_DOFS {
                let value = tensors[element][row * LOCAL_DOFS + col];
                output[element * LOCAL_DOFS * LOCAL_DOFS + col * LOCAL_DOFS + row] =
                    if conjugate { value.conj() } else { value };
            }
        }
    }
    output
}

fn apply_reference(
    tensors: &[[Complex64; LOCAL_DOFS * LOCAL_DOFS]; ELEMENTS],
    input: &[Complex64],
) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); DOFS];
    for element in 0..ELEMENTS {
        for row in 0..LOCAL_DOFS {
            for col in 0..LOCAL_DOFS {
                output[element * LOCAL_DOFS + row] +=
                    tensors[element][row * LOCAL_DOFS + col] * input[element * LOCAL_DOFS + col];
            }
        }
    }
    output
}

fn apply_transpose_reference(
    tensors: &[[Complex64; LOCAL_DOFS * LOCAL_DOFS]; ELEMENTS],
    input: &[Complex64],
    conjugate: bool,
) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); DOFS];
    for element in 0..ELEMENTS {
        for row in 0..LOCAL_DOFS {
            for col in 0..LOCAL_DOFS {
                let value = tensors[element][row * LOCAL_DOFS + col];
                output[element * LOCAL_DOFS + col] += if conjugate { value.conj() } else { value }
                    * input[element * LOCAL_DOFS + row];
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

fn stats_value(stats: CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "library_operations": stats.library_operations,
        "synchronization_calls": stats.synchronization_calls,
        "failed_synchronizations": stats.failed_synchronizations,
        "completion_events": stats.completion_events,
        "completion_waits": stats.completion_waits,
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
    })
}

fn native_admission(action: FemAction) -> Result<(), String> {
    let admission = admit_fem_workload(
        FemWorkload::native_cuda_matrix_free(FemScalar::Complex64, action),
        FemBackendRequest::required(FemBackend::Cuda),
    )
    .map_err(|error| format!("matrix-free CUDA admission failed before assembly: {error}"))?;
    if admission.provenance.actual_backend != FemActualBackend::Cuda
        || admission.provenance.fallback != "none"
    {
        return Err(format!(
            "matrix-free admission did not prove native CUDA: {admission:?}"
        ));
    }
    Ok(())
}

fn run(backend: &CudaBackend) -> Result<(Metrics, CudaRuntimeStatsSnapshot), String> {
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        native_admission(action)?;
    }

    let started = Instant::now();
    let (mesh, points) = build_fixture();
    let space = H1Space::new(PolynomialDegree::P1);
    let assembled =
        assemble_bilinear_checked(&HelmholtzForm::new(WAVENUMBER), &space, &space, &mesh, 2)
            .map_err(|error| format!("production P1/Tri3 Helmholtz assembly failed: {error}"))?;
    let production = production_tensors(&assembled)[0];
    let independent = points.map(independent_tensor);
    let production_reference_error = production
        .iter()
        .flatten()
        .zip(independent.iter().flatten())
        .map(|(production, reference)| (*production - *reference).norm())
        .fold(0.0, f64::max);
    if production_reference_error > ACTION_TOLERANCE {
        return Err(format!(
            "production local tensor differs from independent affine reference: {production_reference_error}"
        ));
    }

    let forward_lhs_host = flatten_tensors(&production);
    let transpose_lhs_host = transpose_tensors(&production, false);
    let hermitian_lhs_host = transpose_tensors(&production, true);
    let input: Vec<_> = (0..DOFS)
        .map(|index| {
            Complex64::new(
                (index as f64 * 0.31 + 0.2).sin(),
                (index as f64 * 0.17).cos(),
            )
        })
        .collect();
    let test: Vec<_> = (0..DOFS)
        .map(|index| {
            Complex64::new(
                (index as f64 * 0.19).cos(),
                (index as f64 * 0.29 + 0.1).sin(),
            )
        })
        .collect();
    let delta: Vec<_> = (0..DOFS)
        .map(|index| Complex64::new((index as f64 * 0.13).sin(), (index as f64 * 0.23).cos()))
        .collect();
    let epsilon = 1.0e-7;
    let perturbed: Vec<_> = input
        .iter()
        .zip(&delta)
        .map(|(input, delta)| *input + epsilon * *delta)
        .collect();
    let forward_reference = apply_reference(&independent, &input);
    let transpose_reference = apply_transpose_reference(&independent, &test, false);
    let hermitian_reference = apply_transpose_reference(&independent, &test, true);
    let delta_reference = apply_reference(&independent, &delta);
    let perturbed_reference = apply_reference(&independent, &perturbed);

    let mut device_forward_lhs = backend
        .raw_alloc::<Complex64>(forward_lhs_host.len())
        .map_err(|error| format!("forward tensor allocation failed: {error}"))?;
    let mut device_transpose_lhs = backend
        .raw_alloc::<Complex64>(transpose_lhs_host.len())
        .map_err(|error| format!("transpose tensor allocation failed: {error}"))?;
    let mut device_hermitian_lhs = backend
        .raw_alloc::<Complex64>(hermitian_lhs_host.len())
        .map_err(|error| format!("Hermitian tensor allocation failed: {error}"))?;
    let mut device_input = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("input allocation failed: {error}"))?;
    let mut device_test = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("test allocation failed: {error}"))?;
    let mut device_delta = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("delta allocation failed: {error}"))?;
    let mut device_perturbed = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("perturbed allocation failed: {error}"))?;
    let mut device_forward = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("forward output allocation failed: {error}"))?;
    let mut device_transpose = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("transpose output allocation failed: {error}"))?;
    let mut device_hermitian = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("Hermitian output allocation failed: {error}"))?;
    let mut device_delta_output = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("delta output allocation failed: {error}"))?;
    let mut device_perturbed_output = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("perturbed output allocation failed: {error}"))?;
    for (host, device, label) in [
        (&forward_lhs_host, &mut device_forward_lhs, "forward tensor"),
        (
            &transpose_lhs_host,
            &mut device_transpose_lhs,
            "transpose tensor",
        ),
        (
            &hermitian_lhs_host,
            &mut device_hermitian_lhs,
            "Hermitian tensor",
        ),
        (&input, &mut device_input, "input"),
        (&test, &mut device_test, "test"),
        (&delta, &mut device_delta, "delta"),
        (&perturbed, &mut device_perturbed, "perturbed"),
    ] {
        backend
            .try_copy_h2d(host, device)
            .map_err(|error| format!("{label} upload failed: {error}"))?;
    }

    let initial_baseline = backend.begin_runtime_stats();
    backend
        .try_batched_local_contract(
            &device_forward_lhs,
            &device_input,
            &mut device_forward,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("native matrix-free forward action failed: {error}"))?;
    backend
        .try_batched_local_contract(
            &device_transpose_lhs,
            &device_test,
            &mut device_transpose,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("native matrix-free transpose action failed: {error}"))?;
    backend
        .try_batched_local_contract(
            &device_hermitian_lhs,
            &device_test,
            &mut device_hermitian,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("native matrix-free Hermitian action failed: {error}"))?;
    backend
        .try_batched_local_contract(
            &device_forward_lhs,
            &device_delta,
            &mut device_delta_output,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("native matrix-free Taylor delta action failed: {error}"))?;
    backend
        .try_batched_local_contract(
            &device_forward_lhs,
            &device_perturbed,
            &mut device_perturbed_output,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("native matrix-free Taylor perturbed action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("initial matrix-free synchronization failed: {error}"))?;
    let initial_stats = backend.runtime_stats_since(initial_baseline);

    let mut forward = vec![Complex64::default(); DOFS];
    let mut transpose = vec![Complex64::default(); DOFS];
    let mut hermitian = vec![Complex64::default(); DOFS];
    let mut delta_output = vec![Complex64::default(); DOFS];
    let mut perturbed_output = vec![Complex64::default(); DOFS];
    for (device, host, label) in [
        (&device_forward, &mut forward, "forward"),
        (&device_transpose, &mut transpose, "transpose"),
        (&device_hermitian, &mut hermitian, "Hermitian"),
        (&device_delta_output, &mut delta_output, "Taylor delta"),
        (
            &device_perturbed_output,
            &mut perturbed_output,
            "Taylor perturbed",
        ),
    ] {
        backend
            .try_copy_d2h(device, host)
            .map_err(|error| format!("{label} download failed: {error}"))?;
    }
    let forward_error = max_relative_error(&forward, &forward_reference);
    let transpose_error = max_relative_error(&transpose, &transpose_reference);
    let hermitian_error = max_relative_error(&hermitian, &hermitian_reference);
    let lhs = dotc(&forward, &test);
    let rhs = dotc(&input, &hermitian);
    let inner_product_error = (lhs - rhs).norm() / lhs.norm().max(1.0e-30);
    let taylor_expected: Vec<_> = forward
        .iter()
        .zip(&delta_output)
        .map(|(forward, delta)| *forward + epsilon * *delta)
        .collect();
    let taylor_error = max_relative_error(&perturbed_output, &taylor_expected);
    if forward_error > ACTION_TOLERANCE
        || transpose_error > ACTION_TOLERANCE
        || hermitian_error > ACTION_TOLERANCE
        || inner_product_error > ACTION_TOLERANCE
        || taylor_error > TAYLOR_TOLERANCE
        || max_relative_error(&delta_output, &delta_reference) > ACTION_TOLERANCE
        || max_relative_error(&perturbed_output, &perturbed_reference) > ACTION_TOLERANCE
        || initial_stats.kernel_launches < 5
    {
        return Err(format!(
            "initial matrix-free parity gates failed: forward={forward_error}, transpose={transpose_error}, hermitian={hermitian_error}, inner={inner_product_error}, taylor={taylor_error}, stats={initial_stats:?}"
        ));
    }

    let warm_baseline = backend.begin_runtime_stats();
    backend
        .try_batched_local_contract(
            &device_forward_lhs,
            &device_input,
            &mut device_forward,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("warm forward action failed: {error}"))?;
    backend
        .try_batched_local_contract(
            &device_transpose_lhs,
            &device_test,
            &mut device_transpose,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("warm transpose action failed: {error}"))?;
    backend
        .try_batched_local_contract(
            &device_hermitian_lhs,
            &device_test,
            &mut device_hermitian,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("warm Hermitian action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("warm matrix-free synchronization failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_baseline);
    backend
        .try_copy_d2h(&device_forward, &mut forward)
        .map_err(|error| format!("warm forward download failed: {error}"))?;
    backend
        .try_copy_d2h(&device_transpose, &mut transpose)
        .map_err(|error| format!("warm transpose download failed: {error}"))?;
    backend
        .try_copy_d2h(&device_hermitian, &mut hermitian)
        .map_err(|error| format!("warm Hermitian download failed: {error}"))?;
    let warm_forward_error = max_relative_error(&forward, &forward_reference);
    let warm_transpose_error = max_relative_error(&transpose, &transpose_reference);
    let warm_hermitian_error = max_relative_error(&hermitian, &hermitian_reference);

    let sentinel = vec![Complex64::new(17.0, -19.0); DOFS];
    backend
        .try_copy_h2d(&sentinel, &mut device_forward)
        .map_err(|error| format!("failure sentinel upload failed: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = backend.try_batched_local_contract(
        &device_forward_lhs,
        &device_input,
        &mut device_forward,
        ELEMENTS,
        LOCAL_DOFS,
        LOCAL_DOFS,
        1,
    );
    let failure_typed = matches!(failure, Err(BackendError::Launch { .. }));
    backend.clear_failure();
    backend
        .recover_after_error()
        .map_err(|error| format!("matrix-free launch recovery failed: {error}"))?;
    let mut after_failure = vec![Complex64::default(); DOFS];
    backend
        .try_copy_d2h(&device_forward, &mut after_failure)
        .map_err(|error| format!("failure-preservation download failed: {error}"))?;
    let output_preserved = failure_typed && after_failure == sentinel;
    backend
        .try_batched_local_contract(
            &device_forward_lhs,
            &device_input,
            &mut device_forward,
            ELEMENTS,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .map_err(|error| format!("matrix-free retry action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("retry synchronization failed: {error}"))?;
    backend
        .try_copy_d2h(&device_forward, &mut forward)
        .map_err(|error| format!("retry download failed: {error}"))?;
    let retry_error = max_relative_error(&forward, &forward_reference);

    let bad_lhs = backend
        .raw_alloc::<Complex64>(forward_lhs_host.len() - 1)
        .map_err(|error| format!("malformed lhs allocation failed: {error}"))?;
    let mut malformed_output = backend
        .raw_alloc::<Complex64>(DOFS)
        .map_err(|error| format!("malformed output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&sentinel, &mut malformed_output)
        .map_err(|error| format!("malformed sentinel upload failed: {error}"))?;
    let malformed_error = backend.try_batched_local_contract(
        &bad_lhs,
        &device_input,
        &mut malformed_output,
        ELEMENTS,
        LOCAL_DOFS,
        LOCAL_DOFS,
        1,
    );
    let malformed_rejected = matches!(malformed_error, Err(BackendError::InvalidArgument { .. }));
    let mut malformed_after = vec![Complex64::default(); DOFS];
    backend
        .try_copy_d2h(&malformed_output, &mut malformed_after)
        .map_err(|error| format!("malformed preservation download failed: {error}"))?;
    let malformed_output_preserved = malformed_after == sentinel;

    let empty_lhs = backend
        .raw_alloc::<Complex64>(0)
        .map_err(|error| format!("empty lhs allocation failed: {error}"))?;
    let empty_rhs = backend
        .raw_alloc::<Complex64>(0)
        .map_err(|error| format!("empty rhs allocation failed: {error}"))?;
    let mut empty_output = backend
        .raw_alloc::<Complex64>(0)
        .map_err(|error| format!("empty output allocation failed: {error}"))?;
    let empty_batch_accepted = backend
        .try_batched_local_contract(
            &empty_lhs,
            &empty_rhs,
            &mut empty_output,
            0,
            LOCAL_DOFS,
            LOCAL_DOFS,
            1,
        )
        .is_ok();

    let metrics = Metrics {
        production_reference_error,
        forward_error,
        transpose_error,
        hermitian_error,
        inner_product_error,
        taylor_error,
        warm_forward_error,
        warm_transpose_error,
        warm_hermitian_error,
        retry_error,
        warm_kernel_launches: warm_stats.kernel_launches,
        warm_h2d_operations: warm_stats.transfer.h2d_operations,
        warm_allocation_operations: warm_stats.allocation_operations,
        warm_staging_allocation_operations: warm_stats.staging_allocation_operations,
        failure_typed,
        output_preserved,
        retry_succeeded: retry_error <= ACTION_TOLERANCE,
        malformed_rejected,
        malformed_output_preserved,
        empty_batch_accepted,
        elapsed_ms: started.elapsed().as_millis() as u64,
    };
    if metrics.warm_forward_error > ACTION_TOLERANCE
        || metrics.warm_transpose_error > ACTION_TOLERANCE
        || metrics.warm_hermitian_error > ACTION_TOLERANCE
        || metrics.warm_kernel_launches < 3
        || metrics.warm_h2d_operations != 0
        || metrics.warm_allocation_operations != 0
        || metrics.warm_staging_allocation_operations != 0
        || !metrics.failure_typed
        || !metrics.output_preserved
        || !metrics.retry_succeeded
        || !metrics.malformed_rejected
        || !metrics.malformed_output_preserved
        || !metrics.empty_batch_accepted
    {
        return Err(format!("matrix-free CUDA gates failed: {metrics:?}"));
    }
    Ok((metrics, backend.runtime_stats()))
}

fn artifact(metrics: &Metrics, stats: CudaRuntimeStatsSnapshot) -> Value {
    let capability_rows: Vec<_> = fem_cuda_capability_rows()
        .into_iter()
        .filter(|row| row.mode == sonium_fem::FemAssemblyMode::MatrixFree)
        .map(|row| serde_json::to_value(row).expect("matrix-free capability row serializes"))
        .collect();
    json!({
        "schema": SCHEMA,
        "status": "passed",
        "case_id": CASE_ID,
        "scope": "native Linux x86_64 required CUDA SM120; Windows/Arm64/AVX512 waived",
        "claim_class": "production H1/Tri3/P1 Helmholtz matrix-free batched local action",
        "cell": {
            "space": "H1",
            "element": "Tri3",
            "order": "P1",
            "physics": "Helmholtz",
            "mode": "matrix-free",
            "scalar": "Complex64",
            "actions": ["forward", "transpose", "hermitian"],
        },
        "assembly": {
            "production": "HelmholtzForm + assemble_bilinear_checked",
            "execution": "Backend::try_batched_local_contract native CUDA kernel",
            "element_tensor_setup": "host production assembly; no global CSR upload",
            "independent_reference": "affine P1/Tri3 stiffness plus consistent mass analytic oracle",
        },
        "dimensions": {
            "elements": ELEMENTS,
            "local_dofs": LOCAL_DOFS,
            "batches": ELEMENTS,
            "output_entries": DOFS,
            "tail_case": true,
            "tail_output_entries": DOFS,
            "tail_not_multiple_of_cuda_block_256": true,
        },
        "metrics": {
            "production_reference_error": metrics.production_reference_error,
            "forward_error": metrics.forward_error,
            "transpose_error": metrics.transpose_error,
            "hermitian_error": metrics.hermitian_error,
            "inner_product_error": metrics.inner_product_error,
            "taylor_error": metrics.taylor_error,
            "warm_forward_error": metrics.warm_forward_error,
            "warm_transpose_error": metrics.warm_transpose_error,
            "warm_hermitian_error": metrics.warm_hermitian_error,
            "retry_error": metrics.retry_error,
            "warm_kernel_launches": metrics.warm_kernel_launches,
            "warm_h2d_operations": metrics.warm_h2d_operations,
            "warm_allocation_operations": metrics.warm_allocation_operations,
            "warm_staging_allocation_operations": metrics.warm_staging_allocation_operations,
            "failure_typed": metrics.failure_typed,
            "output_preserved": metrics.output_preserved,
            "retry_succeeded": metrics.retry_succeeded,
            "malformed_rejected": metrics.malformed_rejected,
            "malformed_output_preserved": metrics.malformed_output_preserved,
            "empty_batch_accepted": metrics.empty_batch_accepted,
        },
        "thresholds": {
            "production_reference_error": ACTION_TOLERANCE,
            "forward_error": ACTION_TOLERANCE,
            "transpose_error": ACTION_TOLERANCE,
            "hermitian_error": ACTION_TOLERANCE,
            "inner_product_error": ACTION_TOLERANCE,
            "taylor_error": TAYLOR_TOLERANCE,
            "warm_forward_error": ACTION_TOLERANCE,
            "warm_transpose_error": ACTION_TOLERANCE,
            "warm_hermitian_error": ACTION_TOLERANCE,
            "retry_error": ACTION_TOLERANCE,
            "warm_kernel_launches": 3,
            "warm_h2d_operations": 0,
            "warm_allocation_operations": 0,
            "warm_staging_allocation_operations": 0,
            "failure_typed": 1,
            "output_preserved": 1,
            "retry_succeeded": 1,
            "malformed_rejected": 1,
            "malformed_output_preserved": 1,
            "empty_batch_accepted": 1,
        },
        "threshold_directions": {
            "production_reference_error": "maximum",
            "forward_error": "maximum",
            "transpose_error": "maximum",
            "hermitian_error": "maximum",
            "inner_product_error": "maximum",
            "taylor_error": "maximum",
            "warm_forward_error": "maximum",
            "warm_transpose_error": "maximum",
            "warm_hermitian_error": "maximum",
            "retry_error": "maximum",
            "warm_kernel_launches": "minimum",
            "warm_h2d_operations": "maximum",
            "warm_allocation_operations": "maximum",
            "warm_staging_allocation_operations": "maximum",
            "failure_typed": "minimum",
            "output_preserved": "minimum",
            "retry_succeeded": "minimum",
            "malformed_rejected": "minimum",
            "malformed_output_preserved": "minimum",
            "empty_batch_accepted": "minimum",
        },
        "runtime": stats_value(stats),
        "warm_residency": {
            "local_tensors": "device-resident",
            "local_vectors": "device-resident",
            "actions": 3,
            "h2d_operations": metrics.warm_h2d_operations,
            "allocation_operations": metrics.warm_allocation_operations,
            "staging_allocation_operations": metrics.warm_staging_allocation_operations,
        },
        "dispatch": {
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "fallback": "none",
            "admission_before_assembly": true,
            "native_batched_local_kernel": true,
            "matrix_free_global_csr": false,
        },
        "failure_contracts": [
            {
                "id": "launch.output_preservation_and_retry",
                "status": "typed_failure_recovered",
                "stage": "launch",
                "error": "BackendError::Launch",
                "output_preserved": metrics.output_preserved,
                "retry_succeeded": metrics.retry_succeeded,
                "fallback": "none",
            },
            {
                "id": "malformed_local_tensor.shape",
                "status": "typed_rejection",
                "stage": "action_admission",
                "error": "BackendError::InvalidArgument",
                "rejected": metrics.malformed_rejected,
                "output_preserved": metrics.malformed_output_preserved,
                "fallback": "none",
            },
            {
                "id": "empty_batch.noop",
                "status": "accepted_noop",
                "stage": "action_admission",
                "accepted": metrics.empty_batch_accepted,
                "fallback": "none",
            },
        ],
        "capability_rows": capability_rows,
        "unsupported_scope": [
            "element geometry/quadrature integration remains host setup",
            "partial assembly, device-resident constraints, and non-Tri3 or higher-order matrix-free cells remain unsupported",
            "this packet does not claim a general FEM matrix-free dispatcher or global solve",
        ],
        "hardware": cuda_hardware_provenance(),
        "workload_elapsed_ms": metrics.elapsed_ms.max(1),
    })
}

fn main() {
    assert_eq!(std::env::consts::OS, "linux", "FEM1 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "FEM1 requires native x86_64"
    );
    let backend = required_cuda_backend().expect("required SM120 CUDA backend");
    let (metrics, stats) = run(&backend).expect("FEM1 matrix-free CUDA gates");
    let payload = serde_json::to_vec_pretty(&artifact(&metrics, stats))
        .expect("serialize FEM1 matrix-free CUDA artifact");
    let (path, checksum) = write_qa_artifact("fem1_matrix_free_cuda.json", &payload)
        .unwrap_or_else(|error| {
            eprintln!("FAILED: write FEM1 matrix-free CUDA artifact: {error}");
            std::process::exit(1)
        });
    println!("artifact={path} checksum={checksum}");
    println!("SUCCESS");
}
