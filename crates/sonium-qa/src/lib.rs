#![allow(clippy::needless_range_loop)]

//! Integration test harness for sonium solvers, BEM, and FEM
//!
//! This crate provides integration tests that validate algorithm accuracy
//! against analytical solutions where available, or self-consistency checks
//! (symmetry, conservation, FD validation) otherwise.
//!
//! Acceptance thresholds are case-specific. Many analytical gates target <1%;
//! coarse discretizations and selected physical observables have documented
//! wider bounds. See the Wolfram confidence matrix for the actual evidence.

use num_traits::Float;
use num_traits::FromPrimitive;
use num_traits::Zero;
use rand::Rng;
use sha2::{Digest, Sha256};
use sonium_solvers::{ComplexField, CsrBuilder, CsrMatrix};
use std::f64::consts::PI;

pub mod analytical;
pub mod fem_physics_3d;
pub mod fixtures;
#[cfg(feature = "cuda")]
pub mod pcfan_cuda_adjoint;
pub mod rank;
pub mod result;
pub mod subprocess;
pub mod validation;
pub mod wolfram_audio;

pub use analytical::*;
pub use fixtures::*;
pub use result::{
    QA_RESULT_SCHEMA_ID, QA_RESULT_SCHEMA_VERSION, QaArtifactProvenance, QaBackendOutcome,
    QaContextReuse, QaEvidence, QaExecutionCounts, QaFailureStage, QaHardwareProvenance,
    QaMemoryEvidence, QaResult, QaRuntimeCounters, QaRuntimeEvidence, QaStatus,
    QaThresholdDirection, is_supported_schema_version,
};

/// Read a non-empty provenance value from the environment.
fn provenance_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// Collect host/compiler metadata shared by native CPU and accelerator cases.
pub fn host_hardware_provenance() -> QaHardwareProvenance {
    let cpu_model = provenance_env("SONIUM_QA_CPU_MODEL").or_else(|| {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    line.split_once(':')
                        .filter(|(key, _)| key.trim() == "model name" || key.trim() == "Hardware")
                        .map(|(_, value)| value.trim().to_owned())
                })
            })
    });
    let microcode = provenance_env("SONIUM_QA_MICROCODE").or_else(|| {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    line.split_once(':')
                        .filter(|(key, _)| key.trim() == "microcode")
                        .map(|(_, value)| value.trim().to_owned())
                })
            })
    });
    let compiler = provenance_env("SONIUM_QA_COMPILER").or_else(|| {
        std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    });

    QaHardwareProvenance {
        cpu_model,
        microcode,
        os: Some(
            provenance_env("SONIUM_QA_OS")
                .unwrap_or_else(|| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)),
        ),
        compiler,
        target_features: provenance_env("SONIUM_QA_TARGET_FEATURES"),
        runtime_dispatch: provenance_env("SONIUM_QA_RUNTIME_DISPATCH"),
        thread_policy: Some(
            provenance_env("SONIUM_QA_THREAD_POLICY").unwrap_or_else(|| "unspecified".into()),
        ),
        affinity_policy: Some(
            provenance_env("SONIUM_QA_AFFINITY_POLICY").unwrap_or_else(|| "unspecified".into()),
        ),
        numa_policy: Some(
            provenance_env("SONIUM_QA_NUMA_POLICY").unwrap_or_else(|| "unspecified".into()),
        ),
        deterministic_mode: Some(
            provenance_env("SONIUM_QA_DETERMINISTIC_MODE").unwrap_or_else(|| "unspecified".into()),
        ),
        ..QaHardwareProvenance::default()
    }
}

/// Write a release artifact beneath `SONIUM_OUTPUT_DIR` and return its
/// relative path plus a content checksum suitable for `QaResult` provenance.
pub fn write_qa_artifact(relative: &str, bytes: &[u8]) -> Result<(String, String), String> {
    let directory = std::env::var_os("SONIUM_OUTPUT_DIR")
        .ok_or_else(|| "SONIUM_OUTPUT_DIR is required for release evidence".to_owned())?;
    let path = std::path::Path::new(relative);
    if relative.trim().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(component, std::path::Component::ParentDir)
                || matches!(component, std::path::Component::RootDir)
        })
        || relative.contains('\\')
        || relative.contains(':')
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::CurDir))
    {
        return Err(format!("unsafe QA artifact path {relative:?}"));
    }
    let output = std::path::PathBuf::from(directory).join(path);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create QA artifact directory: {error}"))?;
    }
    std::fs::write(&output, bytes).map_err(|error| format!("write QA artifact: {error}"))?;
    let digest = Sha256::digest(bytes);
    Ok((
        relative.to_owned(),
        format!(
            "sha256:{}",
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ),
    ))
}

/// Add CUDA device/toolkit fields exported by the validated accelerator lane.
#[cfg(feature = "cuda")]
pub fn cuda_hardware_provenance() -> QaHardwareProvenance {
    QaHardwareProvenance {
        device: provenance_env("SONIUM_QA_CUDA_DEVICE_NAME"),
        driver: provenance_env("SONIUM_QA_CUDA_DRIVER_VERSION"),
        toolkit: provenance_env("SONIUM_QA_CUDA_TOOLKIT_VERSION"),
        topology: provenance_env("SONIUM_QA_CUDA_TOPOLOGY"),
        peak_flops: provenance_env("SONIUM_QA_CUDA_PEAK_FLOPS"),
        memory_bandwidth: provenance_env("SONIUM_QA_CUDA_MEMORY_BANDWIDTH"),
        ..host_hardware_provenance()
    }
}

/// Initialize a required CUDA backend, honoring the checked device selector.
///
/// `SONIUM_CUDA_FORCE_FALLBACK=1` is a deliberate fault-injection switch used
/// by CI.  It must turn a required accelerator case into a process failure;
/// it must never silently return a CPU-backed result.
#[cfg(feature = "cuda")]
pub fn required_cuda_backend() -> Result<sonium_compute::CudaBackend, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let ordinal = std::env::var("SONIUM_CUDA_DEVICE_ORDINAL")
        .unwrap_or_else(|_| "0".into())
        .parse::<usize>()
        .map_err(|error| format!("invalid SONIUM_CUDA_DEVICE_ORDINAL: {error}"))?;
    sonium_compute::CudaBackend::new(ordinal)
        .map_err(|error| format!("required CUDA backend initialization failed: {error}"))
}

#[cfg(feature = "cuda")]
pub fn cuda_runtime_evidence(
    stats: sonium_compute::CudaRuntimeStatsSnapshot,
    counts: QaExecutionCounts,
) -> QaRuntimeEvidence {
    QaRuntimeEvidence {
        context_reuse: QaContextReuse::Reused,
        counts,
        counters: QaRuntimeCounters {
            allocations: stats
                .allocation_operations
                .saturating_add(stats.staging_allocation_operations)
                as u64,
            h2d_transfers: stats.transfer.h2d_operations as u64,
            d2h_transfers: stats.transfer.d2h_operations as u64,
            d2d_transfers: stats.transfer.d2d_operations as u64,
            synchronizations: stats.synchronization_calls as u64,
            launches: (stats.kernel_launches + stats.library_operations) as u64,
        },
        memory: QaMemoryEvidence {
            peak_bytes: stats
                .allocation_high_water_bytes
                .saturating_add(stats.staging_high_water_bytes) as u64,
            live_bytes: stats
                .live_allocation_bytes
                .saturating_add(stats.live_staging_bytes) as u64,
        },
        hardware: cuda_hardware_provenance(),
        ..QaRuntimeEvidence::default()
    }
}
pub use validation::*;

pub const REL_TOL: f64 = 1e-2;
pub const ABS_TOL: f64 = 1e-8;
pub const RESIDUAL_TOL: f64 = 1e-8;
pub const GRADIENT_TOL: f64 = 1e-8;
pub const KKT_TOL: f64 = 1e-6;

pub fn relative_error<T: ComplexField>(a: &[T], b: &[T]) -> T::Real
where
    T::Real: Float + Zero,
{
    if a.len() != b.len() || a.is_empty() {
        return T::Real::infinity();
    }

    let mut num_sqr = T::Real::zero();
    let mut den_sqr = T::Real::zero();
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        let diff_sqr = diff.norm_sqr();
        let reference_sqr = b[i].norm_sqr();
        if !diff_sqr.is_finite() || !reference_sqr.is_finite() {
            return T::Real::infinity();
        }
        num_sqr += diff_sqr;
        den_sqr += reference_sqr;
        if !num_sqr.is_finite() || !den_sqr.is_finite() {
            return T::Real::infinity();
        }
    }

    if den_sqr <= T::Real::epsilon() {
        if num_sqr <= T::Real::epsilon() {
            T::Real::zero()
        } else {
            T::Real::infinity()
        }
    } else {
        (num_sqr / den_sqr).sqrt()
    }
}

/// Relative L2 error for a field sampled over weighted surface elements.
///
/// `weights` are typically element areas. Invalid dimensions, non-finite
/// samples, negative weights, and a zero total weight fail closed by returning
/// infinity so a conventional `error <= tolerance` QA gate cannot pass them.
pub fn weighted_relative_l2_error<T: ComplexField>(
    computed: &[T],
    reference: &[T],
    weights: &[T::Real],
) -> T::Real
where
    T::Real: Float + Zero,
{
    if computed.len() != reference.len() || computed.len() != weights.len() || computed.is_empty() {
        return T::Real::infinity();
    }

    let mut numerator = T::Real::zero();
    let mut denominator = T::Real::zero();
    let mut total_weight = T::Real::zero();
    for i in 0..computed.len() {
        let weight = weights[i];
        let difference_sqr = (computed[i] - reference[i]).norm_sqr();
        let reference_sqr = reference[i].norm_sqr();
        if !weight.is_finite()
            || weight < T::Real::zero()
            || !difference_sqr.is_finite()
            || !reference_sqr.is_finite()
        {
            return T::Real::infinity();
        }

        numerator += weight * difference_sqr;
        denominator += weight * reference_sqr;
        total_weight += weight;
        if !numerator.is_finite() || !denominator.is_finite() || !total_weight.is_finite() {
            return T::Real::infinity();
        }
    }

    if total_weight <= T::Real::epsilon() {
        return T::Real::infinity();
    }
    if denominator <= T::Real::epsilon() {
        if numerator <= T::Real::epsilon() {
            T::Real::zero()
        } else {
            T::Real::infinity()
        }
    } else {
        (numerator / denominator).sqrt()
    }
}

/// Triangle-area-weighted complex surface-pressure error against the exact
/// rigid-sphere Mie trace at each element's own collocation direction.
///
/// This deliberately compares aligned complex fields. Averaging numerical
/// magnitudes over every triangle and comparing that scalar with a handful of
/// analytical angles can hide phase, quadrature, and directional errors.
pub fn rigid_sphere_surface_relative_l2_error(
    computed: &[num_complex::Complex64],
    elements: &[sonium_bem::core::types::Element],
    wavenumber: f64,
    radius: f64,
    num_terms: usize,
) -> f64 {
    if computed.len() != elements.len()
        || computed.is_empty()
        || !wavenumber.is_finite()
        || wavenumber <= 0.0
        || !radius.is_finite()
        || radius <= 0.0
        || num_terms == 0
    {
        return f64::INFINITY;
    }

    let mut theta = Vec::with_capacity(elements.len());
    let mut weights = Vec::with_capacity(elements.len());
    for element in elements {
        let [x, y, z] = element.center;
        let center_radius = (x * x + y * y + z * z).sqrt();
        if !center_radius.is_finite() || center_radius <= 0.0 || !element.area.is_finite() {
            return f64::INFINITY;
        }
        theta.push((z / center_radius).clamp(-1.0, 1.0).acos());
        weights.push(element.area);
    }

    let reference = sonium_bem::analytical::sphere_scattering_3d(
        wavenumber,
        radius,
        num_terms,
        vec![radius],
        theta,
    )
    .pressure;
    weighted_relative_l2_error(computed, &reference, &weights)
}

pub fn l2_norm<T: ComplexField>(a: &[T]) -> T::Real
where
    T::Real: Float + Zero,
{
    let mut sum = T::Real::zero();
    for i in 0..a.len() {
        sum += a[i].norm_sqr();
    }
    sum.sqrt()
}

pub fn residual_norm<T: ComplexField>(mat: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real
where
    T::Real: Float + Zero,
{
    let ax = mat.matvec(x);
    let mut sum = T::Real::zero();
    for i in 0..b.len() {
        let diff = ax[i] - b[i];
        sum += diff.norm_sqr();
    }
    sum.sqrt()
}

pub fn relative_residual_norm<T: ComplexField>(mat: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real
where
    T::Real: Float + Zero,
{
    let residual = residual_norm(mat, x, b);
    let rhs_norm = l2_norm(b);

    if rhs_norm <= T::Real::epsilon() {
        if residual <= T::Real::epsilon() {
            T::Real::zero()
        } else {
            T::Real::infinity()
        }
    } else {
        residual / rhs_norm
    }
}

pub fn check_symmetry<T>(mat: &CsrMatrix<T>) -> bool
where
    T: ComplexField + PartialEq,
    T::Real: Zero + FromPrimitive,
{
    let n = mat.num_rows;
    if n != mat.num_cols {
        return false;
    }
    let eps: T::Real = T::Real::from_f64(1e-10).unwrap_or(T::Real::zero());
    for i in 0..n {
        for j in 0..n {
            let diff = mat.get(i, j) - mat.get(j, i);
            let norm_diff = diff.norm_sqr();
            if norm_diff > eps * eps {
                return false;
            }
        }
    }
    true
}

pub fn check_spd<T>(mat: &CsrMatrix<T>) -> bool
where
    T: ComplexField,
    T::Real: Float + Zero,
{
    // Necessary screening check only: true SPD validation requires Cholesky or
    // an equivalent factorization. This helper intentionally stays cheap for QA
    // smoke tests.
    let n = mat.num_rows;
    if n != mat.num_cols {
        return false;
    }
    for i in 0..n {
        if mat.get(i, i).re() <= T::Real::zero() {
            return false;
        }
    }
    check_symmetry(mat)
}

pub fn sphere_mesh_points(n_theta: usize, n_phi: usize, radius: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(n_theta * n_phi);
    for i in 0..n_theta {
        let theta = PI * (i as f64) / ((n_theta - 1) as f64);
        for j in 0..n_phi {
            let phi = 2.0 * PI * (j as f64) / (n_phi as f64);
            let x = radius * theta.sin() * phi.cos();
            let y = radius * theta.sin() * phi.sin();
            let z = radius * theta.cos();
            points.push([x, y, z]);
        }
    }
    points
}

pub fn cylinder_mesh_points(n_theta: usize, n_z: usize, radius: f64, height: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(n_theta * n_z);
    for i in 0..n_z {
        let z = -height / 2.0 + height * (i as f64) / ((n_z - 1) as f64);
        for j in 0..n_theta {
            let theta = 2.0 * PI * (j as f64) / (n_theta as f64);
            let x = radius * theta.cos();
            let y = radius * theta.sin();
            points.push([x, y, z]);
        }
    }
    points
}

pub fn tridiagonal_matrix(n: usize, diag: f64, off_diag: f64) -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(n, n);
    for i in 0..n {
        let mut entries = vec![(i, diag)];
        if i > 0 {
            entries.push((i - 1, off_diag));
        }
        if i < n - 1 {
            entries.push((i + 1, off_diag));
        }
        builder.add_row_entries(entries.into_iter());
    }
    builder.finish()
}

pub fn spd_tridiagonal(n: usize) -> CsrMatrix<f64> {
    tridiagonal_matrix(n, 2.0, -1.0)
}

pub fn generate_rhs(n: usize, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut b = vec![0.0; n];
    for i in 0..n {
        b[i] = rng.random();
    }
    b
}

pub fn analytical_1d_laplacian_eigenvalues(n: usize) -> Vec<f64> {
    (1..=n)
        .map(|k| 2.0 - 2.0 * (PI * k as f64 / (n + 1) as f64).cos())
        .collect()
}

pub fn analytical_1d_laplacian_eigenvector(k: usize, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for i in 0..n {
        let x = (i + 1) as f64 * PI / (n + 1) as f64;
        v[i] = (k as f64 * x).sin();
    }
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    for i in 0..n {
        v[i] /= norm;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_error_zero_reference_zero_computed_is_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        let err: f64 = relative_error(&a, &b);
        assert_eq!(err, 0.0);
    }

    #[test]
    fn relative_error_zero_reference_nonzero_computed_is_infinite() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 0.0];
        let err: f64 = relative_error(&a, &b);
        assert!(err.is_infinite());
    }

    #[test]
    fn relative_error_rejects_mismatched_lengths() {
        let err: f64 = relative_error(&[1.0], &[1.0, 2.0]);
        assert!(err.is_infinite());
    }

    #[test]
    fn relative_error_rejects_nonfinite_samples() {
        let nan_err: f64 = relative_error(&[f64::NAN], &[1.0]);
        let infinite_err: f64 = relative_error(&[1.0], &[f64::INFINITY]);
        assert!(nan_err.is_infinite());
        assert!(infinite_err.is_infinite());
    }

    #[test]
    fn weighted_relative_l2_error_uses_surface_weights() {
        use num_complex::Complex64;

        let computed = [Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)];
        let reference = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let weights = [4.0, 1.0];
        let err = weighted_relative_l2_error(&computed, &reference, &weights);

        assert!((err - 0.8_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn weighted_relative_l2_error_rejects_invalid_inputs() {
        use num_complex::Complex64;

        let values = [Complex64::new(1.0, 0.0)];
        assert!(weighted_relative_l2_error(&values, &values, &[]).is_infinite());
        assert!(weighted_relative_l2_error(&values, &values, &[-1.0]).is_infinite());
        assert!(weighted_relative_l2_error(&values, &values, &[f64::NAN]).is_infinite());
    }

    #[test]
    fn rigid_sphere_surface_error_compares_each_element_at_its_own_angle() {
        use sonium_bem::analytical::sphere_scattering_3d;
        use sonium_bem::core::mesh::generators::generate_icosphere_mesh;

        let radius = 0.5;
        let wavenumber = 2.0;
        let mesh = generate_icosphere_mesh(radius, 1);
        let theta: Vec<f64> = mesh
            .elements
            .iter()
            .map(|element| {
                let [x, y, z] = element.center;
                (z / (x * x + y * y + z * z).sqrt()).clamp(-1.0, 1.0).acos()
            })
            .collect();
        let exact = sphere_scattering_3d(wavenumber, radius, 40, vec![radius], theta).pressure;

        let error =
            rigid_sphere_surface_relative_l2_error(&exact, &mesh.elements, wavenumber, radius, 40);
        assert!(error < 1e-14, "aligned exact field error = {error}");
    }

    #[test]
    fn rigid_sphere_surface_error_detects_global_complex_sign_mutation() {
        use sonium_bem::analytical::sphere_scattering_3d;
        use sonium_bem::core::mesh::generators::generate_icosphere_mesh;

        let radius = 0.5;
        let wavenumber = 2.0;
        let mesh = generate_icosphere_mesh(radius, 1);
        let theta: Vec<f64> = mesh
            .elements
            .iter()
            .map(|element| {
                let [x, y, z] = element.center;
                (z / (x * x + y * y + z * z).sqrt()).clamp(-1.0, 1.0).acos()
            })
            .collect();
        let exact = sphere_scattering_3d(wavenumber, radius, 40, vec![radius], theta).pressure;
        let sign_flipped: Vec<_> = exact.iter().map(|value| -*value).collect();

        let error = rigid_sphere_surface_relative_l2_error(
            &sign_flipped,
            &mesh.elements,
            wavenumber,
            radius,
            40,
        );
        assert!(
            error > 1.0,
            "a global complex-sign mutation must exceed every production surface-error budget; got {error}"
        );
    }

    #[test]
    fn rigid_sphere_surface_error_rejects_invalid_physical_reference_inputs() {
        use num_complex::Complex64;
        use sonium_bem::core::mesh::generators::generate_icosphere_mesh;

        let mesh = generate_icosphere_mesh(0.5, 1);
        let computed = vec![Complex64::new(0.0, 0.0); mesh.elements.len()];
        for (wavenumber, radius, num_terms) in [(0.0, 0.5, 20), (1.0, 0.0, 20), (1.0, 0.5, 0)] {
            assert!(
                rigid_sphere_surface_relative_l2_error(
                    &computed,
                    &mesh.elements,
                    wavenumber,
                    radius,
                    num_terms,
                )
                .is_infinite()
            );
        }
    }

    #[test]
    fn relative_residual_norm_normalizes_by_rhs() {
        let mat = spd_tridiagonal(3);
        let x = vec![1.0, 2.0, 1.0];
        let b = mat.matvec(&x);

        let residual: f64 = relative_residual_norm(&mat, &x, &b);
        assert!(residual < 1e-14);
    }

    #[test]
    fn relative_residual_norm_zero_rhs_nonzero_residual_is_infinite() {
        let mat = spd_tridiagonal(2);
        let x = vec![1.0, 0.0];
        let b = vec![0.0, 0.0];

        let residual: f64 = relative_residual_norm(&mat, &x, &b);
        assert!(residual.is_infinite());
    }
}
