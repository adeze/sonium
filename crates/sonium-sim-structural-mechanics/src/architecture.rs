//! Deterministic native CPU evidence for cross-architecture structural replay.

use crate::{
    BracketStudy, BucklingStudy, FemTopology2DStudy, run_fem_topology_2d, run_linear_buckling,
    run_static_bracket, solve_first_bending_mode,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::process::Command;

pub const CROSS_ARCH_STRUCTURAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeStructuralArchitectureMetadata {
    pub target_arch: String,
    pub target_os: String,
    pub compiler: String,
    pub cpu_vendor: String,
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
    pub scalar_dispatch: String,
    pub simd_dispatch: String,
    pub vendor_dispatch: String,
    pub threads: usize,
    pub deterministic_mode: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeStructuralReferencePacket {
    pub schema_version: u32,
    pub metadata: NativeStructuralArchitectureMetadata,
    pub static_topology: [usize; 3],
    pub static_displacement_m: Vec<[f64; 2]>,
    pub static_max_von_mises_pa: f64,
    pub static_strain_energy_j: f64,
    pub static_solver_residual: f64,
    pub static_reaction_force_n: [f64; 2],
    pub static_force_balance_residual_n: f64,
    pub modal_eigenvalue: f64,
    pub modal_frequency_hz: f64,
    pub modal_solver_residual: f64,
    pub buckling_topology: [usize; 2],
    pub buckling_eigenvalue_n: f64,
    pub buckling_residual: f64,
    pub topology_grid: [usize; 2],
    pub topology_final_density: Vec<f64>,
    pub topology_displacement_m: Vec<[f64; 2]>,
    pub topology_max_displacement_m: f64,
    pub topology_aggregated_stress_pa: f64,
    pub topology_compliance_j: f64,
    pub topology_energy_residual: f64,
    pub topology_solver_residual: f64,
    pub canonical_payload_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CrossArchitectureTolerances {
    pub displacement_relative: f64,
    pub stress_relative: f64,
    pub eigen_relative: f64,
    pub buckling_relative: f64,
    pub energy_relative: f64,
    pub residual_absolute: f64,
    pub density_absolute: f64,
}

impl Default for CrossArchitectureTolerances {
    fn default() -> Self {
        Self {
            displacement_relative: 2.0e-11,
            stress_relative: 2.0e-11,
            eigen_relative: 2.0e-10,
            buckling_relative: 2.0e-10,
            energy_relative: 2.0e-11,
            residual_absolute: 1.0e-10,
            density_absolute: 2.0e-11,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossArchitectureComparison {
    pub schema_version: u32,
    pub reference_arch: String,
    pub candidate_arch: String,
    pub topology_equal: bool,
    pub max_displacement_relative_error: f64,
    pub max_stress_relative_error: f64,
    pub max_eigen_relative_error: f64,
    pub buckling_relative_error: f64,
    pub max_energy_relative_error: f64,
    pub max_residual_absolute_error: f64,
    pub max_density_absolute_error: f64,
    pub serialized_schema_compatible: bool,
    pub canonical_payload_sha256_equal: bool,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralArchitectureError {
    Execution(String),
    Serialization(String),
    InvalidPacket(String),
    ThreadPool(String),
}

impl std::fmt::Display for StructuralArchitectureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution(message) => write!(
                formatter,
                "structural reference execution failed: {message}"
            ),
            Self::Serialization(message) => write!(
                formatter,
                "structural evidence serialization failed: {message}"
            ),
            Self::InvalidPacket(message) => write!(
                formatter,
                "invalid structural architecture packet: {message}"
            ),
            Self::ThreadPool(message) => write!(
                formatter,
                "deterministic structural thread pool failed: {message}"
            ),
        }
    }
}

impl std::error::Error for StructuralArchitectureError {}

#[derive(Serialize)]
struct CanonicalPayload<'a> {
    schema_version: u32,
    static_topology: [usize; 3],
    static_displacement_m: &'a [[f64; 2]],
    static_max_von_mises_pa: f64,
    static_strain_energy_j: f64,
    static_solver_residual: f64,
    static_reaction_force_n: [f64; 2],
    static_force_balance_residual_n: f64,
    modal_eigenvalue: f64,
    modal_frequency_hz: f64,
    modal_solver_residual: f64,
    buckling_topology: [usize; 2],
    buckling_eigenvalue_n: f64,
    buckling_residual: f64,
    topology_grid: [usize; 2],
    topology_final_density: &'a [f64],
    topology_displacement_m: &'a [[f64; 2]],
    topology_max_displacement_m: f64,
    topology_aggregated_stress_pa: f64,
    topology_compliance_j: f64,
    topology_energy_residual: f64,
    topology_solver_residual: f64,
}

fn cpuinfo_value(field: &str) -> Option<String> {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == field).then(|| value.trim().to_string())
            })
        })
}

#[cfg(target_os = "macos")]
fn command_value(program: &str, arguments: &[&str]) -> Option<String> {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty())
}

fn cpu_identity() -> (String, String) {
    #[cfg(target_os = "linux")]
    {
        let vendor = cpuinfo_value(if cfg!(target_arch = "x86_64") {
            "vendor_id"
        } else {
            "CPU implementer"
        })
        .unwrap_or_else(|| "unknown-linux-vendor".to_string());
        let model = cpuinfo_value(if cfg!(target_arch = "x86_64") {
            "model name"
        } else {
            "Model"
        })
        .unwrap_or_else(|| "unknown-linux-model".to_string());
        return (vendor, model);
    }
    #[cfg(target_os = "macos")]
    {
        let model = command_value("sysctl", &["-n", "machdep.cpu.brand_string"])
            .or_else(|| command_value("sysctl", &["-n", "hw.model"]))
            .unwrap_or_else(|| "unknown-apple-model".to_string());
        return ("Apple".to_string(), model);
    }
    #[cfg(target_os = "windows")]
    {
        let vendor = std::env::var("PROCESSOR_ARCHITECTURE")
            .unwrap_or_else(|_| "unknown-windows-vendor".to_string());
        let model = std::env::var("PROCESSOR_IDENTIFIER")
            .unwrap_or_else(|_| "unknown-windows-model".to_string());
        return (vendor, model);
    }
    #[allow(unreachable_code)]
    ("unknown-vendor".to_string(), "unknown-model".to_string())
}

fn compiler_version() -> String {
    Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.lines().next().unwrap_or("unknown").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn detected_features() -> Vec<String> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();
        for (name, present) in [
            ("sse2", std::is_x86_feature_detected!("sse2")),
            ("avx", std::is_x86_feature_detected!("avx")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("fma", std::is_x86_feature_detected!("fma")),
        ] {
            if present {
                features.push(name.to_string());
            }
        }
        features
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut features = vec!["neon".to_string()];
        if std::arch::is_aarch64_feature_detected!("sve") {
            features.push("sve".to_string());
        }
        features
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Vec::new()
    }
}

fn metadata() -> NativeStructuralArchitectureMetadata {
    let features = detected_features();
    let (cpu_vendor, cpu_model) = cpu_identity();
    let simd_dispatch = if cfg!(target_arch = "x86_64") && features.iter().any(|f| f == "avx2") {
        "x86_64-avx2-available; structural sparse reference uses scalar f64 kernels"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64-neon-available; structural sparse reference uses scalar f64 kernels"
    } else {
        "scalar"
    };
    NativeStructuralArchitectureMetadata {
        target_arch: std::env::consts::ARCH.to_string(),
        target_os: std::env::consts::OS.to_string(),
        compiler: compiler_version(),
        cpu_vendor,
        cpu_model,
        cpu_features: features,
        scalar_dispatch: "sonium-solvers deterministic scalar f64 sparse reference".to_string(),
        simd_dispatch: simd_dispatch.to_string(),
        vendor_dispatch: "no vendor BLAS/LAPACK dispatch".to_string(),
        threads: 1,
        deterministic_mode: "single-rayon-worker; fixed mesh/order/iteration controls; f64"
            .to_string(),
    }
}

fn canonical_payload(packet: &NativeStructuralReferencePacket) -> CanonicalPayload<'_> {
    CanonicalPayload {
        schema_version: packet.schema_version,
        static_topology: packet.static_topology,
        static_displacement_m: &packet.static_displacement_m,
        static_max_von_mises_pa: packet.static_max_von_mises_pa,
        static_strain_energy_j: packet.static_strain_energy_j,
        static_solver_residual: packet.static_solver_residual,
        static_reaction_force_n: packet.static_reaction_force_n,
        static_force_balance_residual_n: packet.static_force_balance_residual_n,
        modal_eigenvalue: packet.modal_eigenvalue,
        modal_frequency_hz: packet.modal_frequency_hz,
        modal_solver_residual: packet.modal_solver_residual,
        buckling_topology: packet.buckling_topology,
        buckling_eigenvalue_n: packet.buckling_eigenvalue_n,
        buckling_residual: packet.buckling_residual,
        topology_grid: packet.topology_grid,
        topology_final_density: &packet.topology_final_density,
        topology_displacement_m: &packet.topology_displacement_m,
        topology_max_displacement_m: packet.topology_max_displacement_m,
        topology_aggregated_stress_pa: packet.topology_aggregated_stress_pa,
        topology_compliance_j: packet.topology_compliance_j,
        topology_energy_residual: packet.topology_energy_residual,
        topology_solver_residual: packet.topology_solver_residual,
    }
}

fn payload_digest(
    packet: &NativeStructuralReferencePacket,
) -> Result<String, StructuralArchitectureError> {
    let bytes = serde_json::to_vec(&canonical_payload(packet))
        .map_err(|error| StructuralArchitectureError::Serialization(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_packet(
    packet: &NativeStructuralReferencePacket,
) -> Result<(), StructuralArchitectureError> {
    if packet.schema_version != CROSS_ARCH_STRUCTURAL_SCHEMA_VERSION {
        return Err(StructuralArchitectureError::InvalidPacket(
            "unsupported schema version".to_string(),
        ));
    }
    if packet.metadata.target_arch.is_empty()
        || packet.metadata.compiler.is_empty()
        || packet.metadata.cpu_model.is_empty()
        || packet.metadata.scalar_dispatch.is_empty()
        || packet.metadata.simd_dispatch.is_empty()
        || packet.metadata.vendor_dispatch.is_empty()
        || packet.metadata.threads == 0
        || packet.metadata.deterministic_mode.is_empty()
    {
        return Err(StructuralArchitectureError::InvalidPacket(
            "execution metadata is incomplete".to_string(),
        ));
    }
    if payload_digest(packet)? != packet.canonical_payload_sha256 {
        return Err(StructuralArchitectureError::InvalidPacket(
            "canonical payload digest mismatch".to_string(),
        ));
    }
    Ok(())
}

/// Execute the bounded linear, modal, buckling, and topology packet inside a
/// one-worker pool so the recorded deterministic mode is enforced, not merely
/// described.
pub fn run_native_structural_reference_packet()
-> Result<NativeStructuralReferencePacket, StructuralArchitectureError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .map_err(|error| StructuralArchitectureError::ThreadPool(error.to_string()))?;
    pool.install(|| {
        let mut bracket = BracketStudy::introductory_bracket();
        bracket.geometry.nx = 12;
        bracket.geometry.ny = 3;
        let static_result = run_static_bracket(&bracket)
            .map_err(|error| StructuralArchitectureError::Execution(error.to_string()))?;
        let modal = solve_first_bending_mode(&bracket, &static_result)
            .map_err(|error| StructuralArchitectureError::Execution(error.to_string()))?;

        let mut buckling_study = BucklingStudy::introductory();
        buckling_study.geometry.elements = 8;
        let buckling = run_linear_buckling(&buckling_study)
            .map_err(|error| StructuralArchitectureError::Execution(error.to_string()))?;

        let mut topology_study = FemTopology2DStudy::introductory();
        topology_study.geometry.nx = 6;
        topology_study.geometry.ny = 2;
        topology_study.volume_fraction = 0.95;
        topology_study.move_limit = 0.01;
        topology_study.continuation_steps = 1;
        topology_study.iterations_per_stage = 8;
        topology_study.filter_radius_m = 0.02;
        topology_study.minimum_feature_cells = 1;
        topology_study.maximum_overhang_slope = 100.0;
        let topology = run_fem_topology_2d(&topology_study)
            .map_err(|error| StructuralArchitectureError::Execution(error.to_string()))?;

        let mut packet = NativeStructuralReferencePacket {
            schema_version: CROSS_ARCH_STRUCTURAL_SCHEMA_VERSION,
            metadata: metadata(),
            static_topology: [
                static_result.nodes,
                static_result.elements,
                static_result.dofs,
            ],
            static_displacement_m: static_result.displacement,
            static_max_von_mises_pa: static_result.max_von_mises_pa,
            static_strain_energy_j: static_result.strain_energy_j,
            static_solver_residual: static_result.solver_residual,
            static_reaction_force_n: static_result.reaction_force_n,
            static_force_balance_residual_n: static_result.force_balance_residual_n,
            modal_eigenvalue: modal.eigenvalue,
            modal_frequency_hz: modal.first_bending_frequency_hz,
            modal_solver_residual: modal.solver_residual,
            buckling_topology: [buckling.elements, buckling.dofs],
            buckling_eigenvalue_n: buckling.eigenvalue_n,
            buckling_residual: buckling.residual_norm,
            topology_grid: [topology_study.geometry.nx, topology_study.geometry.ny],
            topology_final_density: topology.final_physical_density,
            topology_displacement_m: topology.displacement,
            topology_max_displacement_m: topology.max_displacement_m,
            topology_aggregated_stress_pa: topology.aggregated_stress_pa,
            topology_compliance_j: topology.final_compliance_j,
            topology_energy_residual: topology.energy_balance_relative_error,
            topology_solver_residual: topology.solver_residual,
            canonical_payload_sha256: String::new(),
        };
        packet.canonical_payload_sha256 = payload_digest(&packet)?;
        Ok(packet)
    })
}

fn rel(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

fn max_vector_rel(left: &[[f64; 2]], right: &[[f64; 2]]) -> f64 {
    let max_difference = left
        .iter()
        .zip(right)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max);
    let scale = left
        .iter()
        .chain(right)
        .flat_map(|value| value.iter().map(|component| component.abs()))
        .fold(0.0, f64::max);
    max_difference / scale.max(f64::MIN_POSITIVE)
}

/// Compare an accepted reference packet (normally aarch64) with a native x86
/// packet using explicit budgets for every promoted observable family.
pub fn compare_structural_architecture_packets(
    reference: &NativeStructuralReferencePacket,
    candidate: &NativeStructuralReferencePacket,
    tolerances: CrossArchitectureTolerances,
) -> Result<CrossArchitectureComparison, StructuralArchitectureError> {
    validate_packet(reference)?;
    validate_packet(candidate)?;
    for tolerance in [
        tolerances.displacement_relative,
        tolerances.stress_relative,
        tolerances.eigen_relative,
        tolerances.buckling_relative,
        tolerances.energy_relative,
        tolerances.residual_absolute,
        tolerances.density_absolute,
    ] {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(StructuralArchitectureError::InvalidPacket(
                "cross-architecture tolerances must be finite and non-negative".to_string(),
            ));
        }
    }
    let topology_equal = reference.static_topology == candidate.static_topology
        && reference.buckling_topology == candidate.buckling_topology
        && reference.topology_grid == candidate.topology_grid
        && reference.static_displacement_m.len() == candidate.static_displacement_m.len()
        && reference.topology_final_density.len() == candidate.topology_final_density.len()
        && reference.topology_displacement_m.len() == candidate.topology_displacement_m.len();
    if !topology_equal {
        return Ok(CrossArchitectureComparison {
            schema_version: CROSS_ARCH_STRUCTURAL_SCHEMA_VERSION,
            reference_arch: reference.metadata.target_arch.clone(),
            candidate_arch: candidate.metadata.target_arch.clone(),
            topology_equal: false,
            max_displacement_relative_error: f64::INFINITY,
            max_stress_relative_error: f64::INFINITY,
            max_eigen_relative_error: f64::INFINITY,
            buckling_relative_error: f64::INFINITY,
            max_energy_relative_error: f64::INFINITY,
            max_residual_absolute_error: f64::INFINITY,
            max_density_absolute_error: f64::INFINITY,
            serialized_schema_compatible: true,
            canonical_payload_sha256_equal: false,
            accepted: false,
        });
    }
    let max_displacement_relative_error = max_vector_rel(
        &candidate.static_displacement_m,
        &reference.static_displacement_m,
    )
    .max(max_vector_rel(
        &candidate.topology_displacement_m,
        &reference.topology_displacement_m,
    ))
    .max(rel(
        candidate.topology_max_displacement_m,
        reference.topology_max_displacement_m,
    ));
    let max_stress_relative_error = rel(
        candidate.static_max_von_mises_pa,
        reference.static_max_von_mises_pa,
    )
    .max(rel(
        candidate.topology_aggregated_stress_pa,
        reference.topology_aggregated_stress_pa,
    ));
    let max_eigen_relative_error = rel(candidate.modal_eigenvalue, reference.modal_eigenvalue).max(
        rel(candidate.modal_frequency_hz, reference.modal_frequency_hz),
    );
    let buckling_relative_error = rel(
        candidate.buckling_eigenvalue_n,
        reference.buckling_eigenvalue_n,
    );
    let max_energy_relative_error = rel(
        candidate.static_strain_energy_j,
        reference.static_strain_energy_j,
    )
    .max(rel(
        candidate.topology_compliance_j,
        reference.topology_compliance_j,
    ));
    let max_residual_absolute_error = [
        (candidate.static_solver_residual - reference.static_solver_residual).abs(),
        (candidate.static_force_balance_residual_n - reference.static_force_balance_residual_n)
            .abs(),
        (candidate.modal_solver_residual - reference.modal_solver_residual).abs(),
        (candidate.buckling_residual - reference.buckling_residual).abs(),
        (candidate.topology_energy_residual - reference.topology_energy_residual).abs(),
        (candidate.topology_solver_residual - reference.topology_solver_residual).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max);
    let max_density_absolute_error = candidate
        .topology_final_density
        .iter()
        .zip(&reference.topology_final_density)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let canonical_payload_sha256_equal =
        candidate.canonical_payload_sha256 == reference.canonical_payload_sha256;
    let accepted = max_displacement_relative_error <= tolerances.displacement_relative
        && max_stress_relative_error <= tolerances.stress_relative
        && max_eigen_relative_error <= tolerances.eigen_relative
        && buckling_relative_error <= tolerances.buckling_relative
        && max_energy_relative_error <= tolerances.energy_relative
        && max_residual_absolute_error <= tolerances.residual_absolute
        && max_density_absolute_error <= tolerances.density_absolute;
    Ok(CrossArchitectureComparison {
        schema_version: CROSS_ARCH_STRUCTURAL_SCHEMA_VERSION,
        reference_arch: reference.metadata.target_arch.clone(),
        candidate_arch: candidate.metadata.target_arch.clone(),
        topology_equal,
        max_displacement_relative_error,
        max_stress_relative_error,
        max_eigen_relative_error,
        buckling_relative_error,
        max_energy_relative_error,
        max_residual_absolute_error,
        max_density_absolute_error,
        serialized_schema_compatible: true,
        canonical_payload_sha256_equal,
        accepted,
    })
}
