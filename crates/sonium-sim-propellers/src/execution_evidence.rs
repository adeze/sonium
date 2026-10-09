//! Backend capability and execution evidence for propeller workloads.
//!
//! The propeller crate currently owns a deterministic CPU/f64 application
//! path.  This module makes that boundary machine-readable: device requests
//! are resolved against an explicit capability matrix and unsupported CUDA or
//! Metal requests are represented as typed results rather than silently
//! falling back to a CPU surrogate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::process::Command;
use std::sync::OnceLock;

use crate::provenance::PropellerModelFidelity;

pub const EXECUTION_EVIDENCE_SCHEMA_V1: &str = "sonium.propeller.execution-evidence.v1";

/// Execution backend requested by a propeller workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropellerBackend {
    Cpu,
    Cuda,
    Metal,
}

impl fmt::Display for PropellerBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        })
    }
}

/// Application lane whose capability is being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropellerWorkload {
    SourceSurrogate,
    BladeElementBem,
    VortexLattice,
    FreeWake,
    BodyFittedMeridional,
    AxisymmetricFlow,
    RotatingWallSwirl,
    ObserverFwh,
    Structural,
}

/// Scalar precision requested by a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropellerPrecision {
    F32,
    F64,
}

/// Direction of an operator or sensitivity operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropellerExecutionDirection {
    Forward,
    Transpose,
}

/// Number of operating points or time histories evaluated by one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropellerExecutionBatch {
    Single,
    Multipoint,
    Transient,
}

/// One fully specified capability-matrix query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerCapabilityRequest {
    pub backend: PropellerBackend,
    pub workload: PropellerWorkload,
    pub precision: PropellerPrecision,
    pub mesh_order: u8,
    pub direction: PropellerExecutionDirection,
    pub batch: PropellerExecutionBatch,
}

/// Capability state for one requested cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerCapabilityCell {
    pub request: PropellerCapabilityRequest,
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Explicit capability matrix for the application-level propeller lanes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerCapabilityMatrix {
    pub schema: String,
    pub cells: Vec<PropellerCapabilityCell>,
}

/// Failure returned when a serialized or caller-provided capability matrix
/// does not match the complete application contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropellerCapabilityMatrixError {
    /// The versioned matrix identity is not the current contract.
    InvalidSchema,
    /// The cartesian backend/workload/scalar/order/action/batch set is absent
    /// or contains an unexpected number of cells.
    InvalidCellCount { expected: usize, actual: usize },
    /// A request is duplicated or missing from the matrix.
    DuplicateOrMissingRequest,
    /// A cell's support state or reason differs from the owning adapter's
    /// declared capability policy.
    CapabilityPolicyMismatch,
}

/// A typed result for capability resolution.  There is deliberately no
/// implicit CPU fallback in this enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropellerCapabilityResult {
    Supported(PropellerCapabilityRequest),
    Unsupported(PropellerUnsupportedCapability),
}

/// Why a requested capability cell cannot be executed by this crate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerUnsupportedCapability {
    pub request: PropellerCapabilityRequest,
    pub requested_backend: PropellerBackend,
    pub actual_backend: Option<PropellerBackend>,
    pub fallback_reason: Option<String>,
    pub reason: String,
}

impl PropellerCapabilityMatrix {
    /// Return the current application capability matrix.
    pub fn current() -> Self {
        let backends = [
            PropellerBackend::Cpu,
            PropellerBackend::Cuda,
            PropellerBackend::Metal,
        ];
        let workloads = [
            PropellerWorkload::SourceSurrogate,
            PropellerWorkload::BladeElementBem,
            PropellerWorkload::VortexLattice,
            PropellerWorkload::FreeWake,
            PropellerWorkload::BodyFittedMeridional,
            PropellerWorkload::AxisymmetricFlow,
            PropellerWorkload::RotatingWallSwirl,
            PropellerWorkload::ObserverFwh,
            PropellerWorkload::Structural,
        ];
        let precisions = [PropellerPrecision::F32, PropellerPrecision::F64];
        let directions = [
            PropellerExecutionDirection::Forward,
            PropellerExecutionDirection::Transpose,
        ];
        let batches = [
            PropellerExecutionBatch::Single,
            PropellerExecutionBatch::Multipoint,
            PropellerExecutionBatch::Transient,
        ];

        let mut cells = Vec::with_capacity(
            backends.len()
                * workloads.len()
                * precisions.len()
                * 2
                * directions.len()
                * batches.len(),
        );
        for backend in backends {
            for workload in workloads {
                for precision in precisions {
                    for mesh_order in 1..=2 {
                        for direction in directions {
                            for batch in batches {
                                let request = PropellerCapabilityRequest {
                                    backend,
                                    workload,
                                    precision,
                                    mesh_order,
                                    direction,
                                    batch,
                                };
                                let reason = unsupported_reason(request);
                                cells.push(PropellerCapabilityCell {
                                    request,
                                    supported: reason.is_none(),
                                    reason,
                                });
                            }
                        }
                    }
                }
            }
        }
        Self {
            schema: EXECUTION_EVIDENCE_SCHEMA_V1.into(),
            cells,
        }
    }

    /// Validate the complete matrix before publishing or executing it.
    ///
    /// This checks the full cartesian request domain and recomputes every
    /// support/reason decision from the propeller-owned policy.  In
    /// particular, a hand-edited matrix cannot turn one CUDA/Metal cell into
    /// a CPU fallback or silently omit an action/batch/scalar dimension.
    pub fn validate(&self) -> Result<(), PropellerCapabilityMatrixError> {
        if self.schema != EXECUTION_EVIDENCE_SCHEMA_V1 {
            return Err(PropellerCapabilityMatrixError::InvalidSchema);
        }
        let expected = Self::current();
        if self.cells.len() != expected.cells.len() {
            return Err(PropellerCapabilityMatrixError::InvalidCellCount {
                expected: expected.cells.len(),
                actual: self.cells.len(),
            });
        }
        for expected_cell in expected.cells {
            let matches = self
                .cells
                .iter()
                .filter(|cell| cell.request == expected_cell.request)
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(PropellerCapabilityMatrixError::DuplicateOrMissingRequest);
            }
            let actual = matches[0];
            if actual.supported != expected_cell.supported || actual.reason != expected_cell.reason
            {
                return Err(PropellerCapabilityMatrixError::CapabilityPolicyMismatch);
            }
        }
        Ok(())
    }

    pub fn cell(&self, request: PropellerCapabilityRequest) -> Option<&PropellerCapabilityCell> {
        self.cells.iter().find(|cell| cell.request == request)
    }

    /// Resolve a request without changing its requested backend.
    pub fn resolve(&self, request: PropellerCapabilityRequest) -> PropellerCapabilityResult {
        match self.cell(request) {
            Some(cell) if cell.supported => PropellerCapabilityResult::Supported(request),
            Some(cell) => PropellerCapabilityResult::Unsupported(PropellerUnsupportedCapability {
                request,
                requested_backend: request.backend,
                actual_backend: None,
                fallback_reason: None,
                reason: cell
                    .reason
                    .clone()
                    .unwrap_or_else(|| "capability is not supported".into()),
            }),
            None => PropellerCapabilityResult::Unsupported(PropellerUnsupportedCapability {
                request,
                requested_backend: request.backend,
                actual_backend: None,
                fallback_reason: None,
                reason: "request is outside the declared capability matrix".into(),
            }),
        }
    }
}

fn unsupported_reason(request: PropellerCapabilityRequest) -> Option<String> {
    if request.backend == PropellerBackend::Cuda
        && request.workload == PropellerWorkload::BladeElementBem
        && request.precision == PropellerPrecision::F64
        && request.mesh_order == 1
        && request.direction == PropellerExecutionDirection::Forward
        && matches!(
            request.batch,
            PropellerExecutionBatch::Single | PropellerExecutionBatch::Multipoint
        )
    {
        #[cfg(feature = "cuda")]
        {
            // One bounded production-connected cell is implemented by the
            // optional adapter. It invokes sonium-compute's native Complex64
            // BEM pair kernel on the propeller-owned panel mesh; it does not
            // imply that every propeller phase is device-resident.
            return None;
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Some("the bounded CUDA propeller adapter requires the cuda feature".into());
        }
    }
    if matches!(
        request.backend,
        PropellerBackend::Cuda | PropellerBackend::Metal
    ) {
        return Some(format!(
            "sonium-sim-propellers has no native {} adapter; use the owning FEM/compute capability contract",
            request.backend
        ));
    }
    if request.precision != PropellerPrecision::F64 {
        return Some("the production propeller application path is f64 only".into());
    }
    if request.mesh_order > 1 {
        return Some("the application reference path declares mesh order 1 only".into());
    }
    if request.direction != PropellerExecutionDirection::Forward {
        return Some("the application reference lanes expose forward execution only".into());
    }
    if request.batch == PropellerExecutionBatch::Multipoint
        && !matches!(
            request.workload,
            PropellerWorkload::SourceSurrogate | PropellerWorkload::BladeElementBem
        )
    {
        return Some("this workload has no declared multipoint application adapter".into());
    }
    if request.batch == PropellerExecutionBatch::Transient
        && !matches!(request.workload, PropellerWorkload::FreeWake)
    {
        return Some("this workload has no declared transient application adapter".into());
    }
    None
}

/// Host and build information attached to every production CPU result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerHostMetadata {
    pub compiler: String,
    pub os: String,
    pub kernel: String,
    pub cpu_model: String,
    pub microcode: String,
    pub target: String,
    pub target_features: String,
    pub rayon_threads: usize,
    pub affinity_policy: String,
}

impl PropellerHostMetadata {
    /// Collect best-effort host facts. Unavailable fields are explicit rather
    /// than being omitted or guessed from a successful compilation.
    pub fn collect() -> Self {
        static HOST: OnceLock<PropellerHostMetadata> = OnceLock::new();
        HOST.get_or_init(Self::collect_uncached).clone()
    }

    fn collect_uncached() -> Self {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let cpu_model = cpuinfo_value(&cpuinfo, &["model name", "Hardware"]);
        let microcode = cpuinfo_value(&cpuinfo, &["microcode"]);
        let kernel = uname_release();
        Self {
            compiler: std::env::var("RUSTC_VERSION")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(rustc_version),
            os: std::env::consts::OS.into(),
            kernel,
            cpu_model,
            microcode,
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            target_features: selected_target_features(),
            rayon_threads: rayon::current_num_threads(),
            affinity_policy: std::env::var("SONIUM_AFFINITY_POLICY")
                .unwrap_or_else(|_| "unspecified".into()),
        }
    }
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unavailable".into())
}

fn cpuinfo_value(contents: &str, keys: &[&str]) -> String {
    contents
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            keys.iter()
                .any(|candidate| key.trim().eq_ignore_ascii_case(candidate))
                .then(|| value.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unavailable".into())
}

fn uname_release() -> String {
    #[cfg(unix)]
    {
        Command::new("uname")
            .arg("-r")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "unavailable".into())
    }
    #[cfg(not(unix))]
    {
        "unavailable".into()
    }
}

fn selected_target_features() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();
        if std::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::is_x86_feature_detected!("fma") {
            features.push("fma");
        }
        if std::is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2");
        }
        if features.is_empty() {
            "scalar".into()
        } else {
            features.join(",")
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return "neon".into();
            }
        }
        "scalar".into()
    }
}

/// Execution and provenance facts attached to an optimizer result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerExecutionProvenance {
    pub requested_backend: PropellerBackend,
    pub actual_backend: Option<PropellerBackend>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    pub precision: PropellerPrecision,
    pub scalar_or_simd: String,
    pub fidelity: PropellerModelFidelity,
    pub converged: bool,
    pub cancelled: bool,
    pub device_lost: bool,
    pub allocation_failures: u64,
    pub transfer_count: u64,
    pub synchronization_count: u64,
    pub launch_count: u64,
    pub peak_live_bytes: u64,
    pub seed: String,
    pub input_checksum_sha256: String,
    pub output_checksum_sha256: String,
    pub host: PropellerHostMetadata,
}

/// Machine-readable checksum index for a complete propeller export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerArtifactEvidence {
    pub schema: String,
    pub provenance: PropellerExecutionProvenance,
    pub artifact_checksums_sha256: BTreeMap<String, String>,
}

impl PropellerExecutionProvenance {
    /// Evidence for the current deterministic application surrogate.
    pub fn cpu_surrogate(seed: impl Into<String>, input: &[u8], output: &[u8]) -> Self {
        Self {
            requested_backend: PropellerBackend::Cpu,
            actual_backend: Some(PropellerBackend::Cpu),
            fallback_reason: None,
            precision: PropellerPrecision::F64,
            // The application-level propeller kernels are scalar f64 code;
            // available host features are recorded separately and are not
            // mistaken for an actually selected SIMD implementation.
            scalar_or_simd: "scalar".into(),
            fidelity: PropellerModelFidelity::PanelGeometrySurrogate,
            converged: false,
            cancelled: false,
            device_lost: false,
            allocation_failures: 0,
            transfer_count: 0,
            synchronization_count: 0,
            launch_count: 0,
            peak_live_bytes: 0,
            seed: seed.into(),
            input_checksum_sha256: sha256_hex(input),
            output_checksum_sha256: sha256_hex(output),
            host: PropellerHostMetadata::collect(),
        }
    }

    /// Whether this record is safe to attach to a publishable CPU surrogate.
    pub fn is_publishable_cpu_surrogate(&self) -> bool {
        self.requested_backend == PropellerBackend::Cpu
            && self.actual_backend == Some(PropellerBackend::Cpu)
            && self.fallback_reason.is_none()
            && self.precision == PropellerPrecision::F64
            && self.fidelity == PropellerModelFidelity::PanelGeometrySurrogate
            && !self.cancelled
            && !self.device_lost
            && self.allocation_failures == 0
            && self.valid_sha256(&self.input_checksum_sha256)
            && self.valid_sha256(&self.output_checksum_sha256)
            && !self.seed.trim().is_empty()
            && !self.host.compiler.trim().is_empty()
            && !self.host.os.trim().is_empty()
            && !self.host.target.trim().is_empty()
            && !self.host.target_features.trim().is_empty()
    }

    fn valid_sha256(&self, value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }
}

/// SHA-256 used for machine-readable input, output, and artifact evidence.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_is_explicit_and_never_falls_back_from_a_device_request() {
        let matrix = PropellerCapabilityMatrix::current();
        assert_eq!(matrix.schema, EXECUTION_EVIDENCE_SCHEMA_V1);
        matrix.validate().unwrap();
        assert!(
            matrix
                .cells
                .iter()
                .any(|cell| { cell.request.backend == PropellerBackend::Cuda && !cell.supported })
        );
        assert!(
            matrix
                .cells
                .iter()
                .any(|cell| { cell.request.backend == PropellerBackend::Metal && !cell.supported })
        );

        let request = PropellerCapabilityRequest {
            backend: PropellerBackend::Cuda,
            workload: PropellerWorkload::VortexLattice,
            precision: PropellerPrecision::F64,
            mesh_order: 1,
            direction: PropellerExecutionDirection::Forward,
            batch: PropellerExecutionBatch::Single,
        };
        let PropellerCapabilityResult::Unsupported(result) = matrix.resolve(request) else {
            panic!("CUDA cannot be reported as supported");
        };
        assert_eq!(result.requested_backend, PropellerBackend::Cuda);
        assert_eq!(result.actual_backend, None);
        assert!(result.fallback_reason.is_none());
    }

    #[test]
    fn matrix_validation_rejects_omitted_or_relabelled_cells() {
        let mut missing = PropellerCapabilityMatrix::current();
        missing.cells.pop();
        assert!(matches!(
            missing.validate(),
            Err(PropellerCapabilityMatrixError::InvalidCellCount { .. })
        ));

        let mut relabelled = PropellerCapabilityMatrix::current();
        let cell = relabelled
            .cells
            .iter_mut()
            .find(|cell| cell.request.backend == PropellerBackend::Cuda)
            .unwrap();
        cell.supported = true;
        cell.reason = None;
        assert_eq!(
            relabelled.validate(),
            Err(PropellerCapabilityMatrixError::CapabilityPolicyMismatch)
        );
    }

    #[test]
    fn cpu_surrogate_evidence_has_machine_readable_checksums_and_host_facts() {
        let evidence = PropellerExecutionProvenance::cpu_surrogate(
            "test-seed",
            b"canonical input",
            b"canonical output",
        );
        assert!(evidence.is_publishable_cpu_surrogate());
        assert_eq!(evidence.scalar_or_simd, "scalar");
        assert_eq!(
            evidence.input_checksum_sha256,
            sha256_hex(b"canonical input")
        );
        assert_eq!(
            evidence.output_checksum_sha256,
            sha256_hex(b"canonical output")
        );
        let json = serde_json::to_string(&evidence).unwrap();
        assert!(json.contains("requested_backend"));
        assert!(json.contains("target_features"));
    }
}
