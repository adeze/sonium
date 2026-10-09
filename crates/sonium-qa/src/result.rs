//! Machine-readable result records shared by QA binaries and release tooling.
//!
//! The legacy binaries still print human-friendly `SUCCESS`/`FAILED` markers.
//! New and migrated cases can additionally emit a [`QaResult`] without
//! changing the runner contract; release tooling can then aggregate numerical
//! evidence instead of scraping console text.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

fn usable_provenance(value: Option<&String>) -> bool {
    value.is_some_and(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "unknown" | "unavailable" | "unspecified" | "n/a"
        )
    })
}

pub const QA_RESULT_SCHEMA_VERSION: u32 = 1;
/// Stable identifier for the release-facing QA result wire contract.
pub const QA_RESULT_SCHEMA_ID: &str = "https://sonium.dev/schemas/qa-result-v1.schema.json";

/// Return whether a result can be decoded by this crate's frozen contract.
///
/// Keeping this check next to [`QA_RESULT_SCHEMA_VERSION`] makes version
/// negotiation explicit for runners and release tooling instead of relying
/// on a permissive JSON decoder.
pub const fn is_supported_schema_version(version: u32) -> bool {
    version == QA_RESULT_SCHEMA_VERSION
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QaStatus {
    Passed,
    Failed,
    Skipped,
}

/// Result for one backend attempted by a multi-backend QA case.
///
/// The aggregate [`QaResult`] remains the release decision for the case, while
/// these outcomes preserve which optional backends were actually exercised.
/// An unavailable optional backend is therefore an explicit typed skip rather
/// than an ad-hoc console record or a non-finite metric.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QaBackendOutcome {
    pub backend: String,
    pub status: QaStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Direction used when evaluating a release threshold.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QaThresholdDirection {
    /// The measured value must be greater than or equal to the threshold.
    Minimum,
    /// The measured value must be less than or equal to the threshold.
    Maximum,
}

/// The stage at which a QA case failed.  This is intentionally a closed set:
/// release tooling must be able to distinguish environment/setup failures
/// from a numerical or workload assertion without scraping free-form text.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QaFailureStage {
    Environment,
    Initialization,
    Allocation,
    Transfer,
    Synchronization,
    Launch,
    Execution,
    Validation,
    Artifact,
    Serialization,
}

/// Whether a case reused its accelerator context between measured actions.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QaContextReuse {
    Reused,
    Recreated,
    #[default]
    NotApplicable,
}

/// Counts emitted by one case or one aggregated shard.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QaExecutionCounts {
    pub executed: u64,
    pub passed: u64,
    pub failed: u64,
    pub unsupported: u64,
}

/// Runtime operations observed by a case.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QaRuntimeCounters {
    pub allocations: u64,
    pub h2d_transfers: u64,
    pub d2h_transfers: u64,
    pub d2d_transfers: u64,
    pub synchronizations: u64,
    pub launches: u64,
}

/// Memory watermarks captured during a case.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QaMemoryEvidence {
    pub peak_bytes: u64,
    pub live_bytes: u64,
}

/// Hardware and dispatch provenance for an evidence record.
///
/// The fields are optional for compatibility with older CPU-only records.
/// Accelerator release cases use [`QaResult::release_eligible_for`] to make
/// the relevant subset mandatory.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QaHardwareProvenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toolkit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub microcode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compiler: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_features: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_dispatch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity_policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numa_policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deterministic_mode: Option<String>,
    /// GPU interconnect/topology captured by the accelerator runner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<String>,
    /// Optional runner-supplied device peak used to normalize roofline rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_flops: Option<String>,
    /// Optional runner-supplied device memory bandwidth used to normalize
    /// roofline rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth: Option<String>,
}

/// Typed runtime evidence attached to every newly emitted result.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QaRuntimeEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_stage: Option<QaFailureStage>,
    pub context_reuse: QaContextReuse,
    pub counts: QaExecutionCounts,
    pub counters: QaRuntimeCounters,
    pub memory: QaMemoryEvidence,
    pub hardware: QaHardwareProvenance,
}

/// Source and generation metadata for one result artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QaArtifactProvenance {
    pub source: String,
    pub generation: String,
    pub checksum: String,
    pub version: String,
}

/// Reproducibility metadata shared by release-facing QA records.
///
/// The fields are optional so existing binaries can migrate incrementally,
/// but new evidence-producing cases should populate every field that applies
/// to their reference and execution path.  Checksums use a descriptive string
/// rather than a hard-coded digest algorithm so legacy manifest identifiers
/// such as `analytic` and `not-applicable` remain representable.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct QaEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_checksum: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_checksum: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formulation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_residual: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_reference_error: Option<f64>,
    #[serde(default)]
    pub timings_ms: BTreeMap<String, u64>,
    /// Checksums keyed by the relative paths declared in `QaResult::artifacts`.
    #[serde(default)]
    pub artifact_checksums: BTreeMap<String, String>,
    /// Source, generation recipe, version, and checksum for each artifact.
    #[serde(default)]
    pub artifact_provenance: BTreeMap<String, QaArtifactProvenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QaResult {
    pub schema_version: u32,
    pub case_id: String,
    pub status: QaStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_backend: Option<String>,
    /// Per-backend outcomes for a case that probes more than one backend.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_outcomes: Vec<QaBackendOutcome>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub thresholds: BTreeMap<String, f64>,
    /// Explicit direction for every release threshold.
    ///
    /// Keeping direction separate from the numeric map preserves the compact
    /// legacy JSON shape while making release evaluation unambiguous.
    #[serde(default)]
    pub threshold_directions: BTreeMap<String, QaThresholdDirection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<String>,
    pub elapsed_ms: u64,
    pub git_sha: String,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default)]
    pub artifacts: Vec<String>,
    /// Structured reproducibility metadata for the result and its artifacts.
    #[serde(default)]
    pub evidence: QaEvidence,
    /// Typed runtime/failure evidence.  `default` permits old records to be
    /// read while all records emitted by this crate include the field.
    #[serde(default)]
    pub runtime: QaRuntimeEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl QaResult {
    pub fn passed(case_id: impl Into<String>) -> Self {
        Self::new(case_id, QaStatus::Passed)
    }

    pub fn failed(case_id: impl Into<String>, message: impl Into<String>) -> Self {
        let mut result = Self::new(case_id, QaStatus::Failed);
        result.message = Some(message.into());
        result.runtime.failure_stage = Some(QaFailureStage::Execution);
        result
    }

    pub fn skipped(case_id: impl Into<String>, reason: impl Into<String>) -> Self {
        let mut result = Self::new(case_id, QaStatus::Skipped);
        result.skip_reason = Some(reason.into());
        result
    }

    fn new(case_id: impl Into<String>, status: QaStatus) -> Self {
        Self {
            schema_version: QA_RESULT_SCHEMA_VERSION,
            case_id: case_id.into(),
            status,
            skip_reason: None,
            requested_backend: None,
            actual_backend: None,
            backend_outcomes: Vec::new(),
            metrics: BTreeMap::new(),
            thresholds: BTreeMap::new(),
            threshold_directions: BTreeMap::new(),
            seed: None,
            elapsed_ms: 0,
            git_sha: std::env::var("GIT_SHA").unwrap_or_else(|_| "unavailable".into()),
            environment: BTreeMap::new(),
            artifacts: Vec::new(),
            evidence: QaEvidence::default(),
            runtime: QaRuntimeEvidence::default(),
            message: None,
        }
    }

    pub fn metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    pub fn threshold(mut self, name: impl Into<String>, value: f64) -> Self {
        self.thresholds.insert(name.into(), value);
        self
    }

    /// Declare a lower-bound release threshold.
    pub fn minimum(mut self, name: impl Into<String>, value: f64) -> Self {
        let name = name.into();
        self.thresholds.insert(name.clone(), value);
        self.threshold_directions
            .insert(name, QaThresholdDirection::Minimum);
        self
    }

    /// Declare an upper-bound release threshold.
    pub fn maximum(mut self, name: impl Into<String>, value: f64) -> Self {
        let name = name.into();
        self.thresholds.insert(name.clone(), value);
        self.threshold_directions
            .insert(name, QaThresholdDirection::Maximum);
        self
    }

    pub fn artifact(mut self, path: impl Into<String>) -> Self {
        self.artifacts.push(path.into());
        self
    }

    /// Declare an artifact and its reproducibility checksum together.
    pub fn artifact_with_checksum(
        mut self,
        path: impl Into<String>,
        checksum: impl Into<String>,
    ) -> Self {
        let path = path.into();
        self.artifacts.push(path.clone());
        self.evidence
            .artifact_checksums
            .insert(path, checksum.into());
        self
    }

    /// Declare an artifact together with its complete provenance record.
    pub fn artifact_with_provenance(
        mut self,
        path: impl Into<String>,
        source: impl Into<String>,
        generation: impl Into<String>,
        version: impl Into<String>,
        checksum: impl Into<String>,
    ) -> Self {
        let path = path.into();
        let checksum = checksum.into();
        self.artifacts.push(path.clone());
        self.evidence
            .artifact_checksums
            .insert(path.clone(), checksum.clone());
        self.evidence.artifact_provenance.insert(
            path,
            QaArtifactProvenance {
                source: source.into(),
                generation: generation.into(),
                checksum,
                version: version.into(),
            },
        );
        self
    }

    pub fn input_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.evidence.input_checksum = Some(checksum.into());
        self
    }

    pub fn reference(mut self, reference: impl Into<String>) -> Self {
        self.evidence.reference = Some(reference.into());
        self
    }

    pub fn reference_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.evidence.reference_checksum = Some(checksum.into());
        self
    }

    pub fn dimension(mut self, dimension: impl Into<String>) -> Self {
        self.evidence.dimension = Some(dimension.into());
        self
    }

    pub fn formulation(mut self, formulation: impl Into<String>) -> Self {
        self.evidence.formulation = Some(formulation.into());
        self
    }

    pub fn tolerance(mut self, tolerance: impl Into<String>) -> Self {
        self.evidence.tolerance = Some(tolerance.into());
        self
    }

    pub fn memory_bytes(mut self, memory_bytes: u64) -> Self {
        self.evidence.memory_bytes = Some(memory_bytes);
        self
    }

    pub fn physical_residual(mut self, residual: f64) -> Self {
        self.evidence.physical_residual = Some(residual);
        self
    }

    pub fn physical_reference_error(mut self, error: f64) -> Self {
        self.evidence.physical_reference_error = Some(error);
        self
    }

    pub fn runtime_evidence(mut self, runtime: QaRuntimeEvidence) -> Self {
        self.runtime = runtime;
        self
    }

    pub fn failure_stage(mut self, stage: QaFailureStage) -> Self {
        self.runtime.failure_stage = Some(stage);
        self
    }

    pub fn context_reuse(mut self, context_reuse: QaContextReuse) -> Self {
        self.runtime.context_reuse = context_reuse;
        self
    }

    pub fn execution_counts(mut self, counts: QaExecutionCounts) -> Self {
        self.runtime.counts = counts;
        self
    }

    pub fn runtime_counters(mut self, counters: QaRuntimeCounters) -> Self {
        self.runtime.counters = counters;
        self
    }

    pub fn memory_evidence(mut self, memory: QaMemoryEvidence) -> Self {
        self.runtime.memory = memory;
        self
    }

    pub fn hardware_provenance(mut self, hardware: QaHardwareProvenance) -> Self {
        self.runtime.hardware = hardware;
        self
    }

    pub fn timing_ms(mut self, name: impl Into<String>, elapsed_ms: u64) -> Self {
        self.evidence.timings_ms.insert(name.into(), elapsed_ms);
        self
    }

    pub fn requested_backend(mut self, backend: impl Into<String>) -> Self {
        self.requested_backend = Some(backend.into());
        self
    }

    pub fn actual_backend(mut self, backend: impl Into<String>) -> Self {
        self.actual_backend = Some(backend.into());
        self
    }

    /// Record a successful backend attempt.
    pub fn backend_passed(mut self, backend: impl Into<String>) -> Self {
        self.backend_outcomes.push(QaBackendOutcome {
            backend: backend.into(),
            status: QaStatus::Passed,
            reason: None,
        });
        self
    }

    /// Record a backend attempt that failed before producing an accepted
    /// result.  Required accelerator cases use this to preserve the exact
    /// dispatch failure in their machine-readable evidence.
    pub fn backend_failed(mut self, backend: impl Into<String>, reason: impl Into<String>) -> Self {
        self.backend_outcomes.push(QaBackendOutcome {
            backend: backend.into(),
            status: QaStatus::Failed,
            reason: Some(reason.into()),
        });
        self
    }

    /// Record an unavailable optional backend without changing the aggregate
    /// case status. Required backend requests must still fail at the runner.
    pub fn backend_skipped(
        mut self,
        backend: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        self.backend_outcomes.push(QaBackendOutcome {
            backend: backend.into(),
            status: QaStatus::Skipped,
            reason: Some(reason.into()),
        });
        self
    }

    pub fn seed(mut self, seed: impl Into<String>) -> Self {
        self.seed = Some(seed.into());
        self
    }

    pub fn elapsed_ms(mut self, elapsed_ms: u64) -> Self {
        self.elapsed_ms = elapsed_ms;
        self
    }

    /// Validate the release-facing invariants before serializing a record.
    pub fn validate(&self) -> Result<(), String> {
        if !is_supported_schema_version(self.schema_version) {
            return Err(format!(
                "unsupported QA result schema version {}",
                self.schema_version
            ));
        }
        if self.case_id.trim().is_empty() {
            return Err("QA result case_id is empty".into());
        }
        if self.git_sha.trim().is_empty() {
            return Err("QA result git_sha is empty".into());
        }
        match self.status {
            QaStatus::Skipped
                if self
                    .skip_reason
                    .as_deref()
                    .is_none_or(|reason| reason.trim().is_empty()) =>
            {
                return Err("skipped QA result must include skip_reason".into());
            }
            QaStatus::Passed | QaStatus::Failed if self.skip_reason.is_some() => {
                return Err("only skipped QA results may include skip_reason".into());
            }
            _ => {}
        }
        if self.status == QaStatus::Failed
            && self
                .message
                .as_deref()
                .is_none_or(|message| message.trim().is_empty())
        {
            return Err("failed QA result must include message".into());
        }
        if self.status == QaStatus::Passed && self.runtime.failure_stage.is_some() {
            return Err("passed QA result must not include failure_stage".into());
        }
        if self
            .requested_backend
            .iter()
            .chain(self.actual_backend.iter())
            .any(|backend| backend.trim().is_empty())
        {
            return Err("QA result backends must be non-empty when provided".into());
        }
        let mut backend_names = std::collections::BTreeSet::new();
        for outcome in &self.backend_outcomes {
            if outcome.backend.trim().is_empty() {
                return Err("QA backend outcome names must be non-empty".into());
            }
            if !backend_names.insert(&outcome.backend) {
                return Err(format!(
                    "QA backend outcome is duplicated for {:?}",
                    outcome.backend
                ));
            }
            match outcome.status {
                QaStatus::Passed => {
                    if outcome.reason.is_some() {
                        return Err(format!(
                            "passed QA backend outcome {:?} must not include a reason",
                            outcome.backend
                        ));
                    }
                }
                QaStatus::Failed | QaStatus::Skipped
                    if outcome
                        .reason
                        .as_deref()
                        .is_none_or(|reason| reason.trim().is_empty()) =>
                {
                    return Err(format!(
                        "non-passing QA backend outcome {:?} must include a reason",
                        outcome.backend
                    ));
                }
                QaStatus::Failed | QaStatus::Skipped => {}
            }
        }
        if self
            .seed
            .as_deref()
            .is_some_and(|seed| seed.trim().is_empty())
        {
            return Err("QA result seed must be non-empty when provided".into());
        }
        if [
            self.evidence.input_checksum.as_deref(),
            self.evidence.reference.as_deref(),
            self.evidence.reference_checksum.as_deref(),
            self.evidence.dimension.as_deref(),
            self.evidence.formulation.as_deref(),
            self.evidence.tolerance.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|value| value.trim().is_empty())
        {
            return Err("QA evidence fields must be non-empty when provided".into());
        }
        if self
            .evidence
            .physical_residual
            .into_iter()
            .chain(self.evidence.physical_reference_error)
            .any(|value| !value.is_finite())
        {
            return Err("physical QA evidence must be finite when provided".into());
        }
        if self.runtime.memory.live_bytes > self.runtime.memory.peak_bytes {
            return Err("QA live memory cannot exceed peak memory".into());
        }
        if self.runtime.counts.passed > self.runtime.counts.executed
            || self.runtime.counts.failed > self.runtime.counts.executed
            || self.runtime.counts.unsupported > self.runtime.counts.executed
        {
            return Err("QA execution counts cannot exceed executed count".into());
        }
        if self
            .runtime
            .hardware
            .device
            .iter()
            .chain(self.runtime.hardware.driver.iter())
            .chain(self.runtime.hardware.toolkit.iter())
            .chain(self.runtime.hardware.cpu_model.iter())
            .chain(self.runtime.hardware.microcode.iter())
            .chain(self.runtime.hardware.os.iter())
            .chain(self.runtime.hardware.compiler.iter())
            .chain(self.runtime.hardware.target_features.iter())
            .chain(self.runtime.hardware.runtime_dispatch.iter())
            .chain(self.runtime.hardware.thread_policy.iter())
            .chain(self.runtime.hardware.affinity_policy.iter())
            .chain(self.runtime.hardware.numa_policy.iter())
            .chain(self.runtime.hardware.deterministic_mode.iter())
            .chain(self.runtime.hardware.topology.iter())
            .chain(self.runtime.hardware.peak_flops.iter())
            .chain(self.runtime.hardware.memory_bandwidth.iter())
            .any(|value| value.trim().is_empty())
        {
            return Err("QA hardware provenance fields must be non-empty when provided".into());
        }
        if self
            .evidence
            .timings_ms
            .keys()
            .any(|name| name.trim().is_empty())
        {
            return Err("QA evidence timing names must be non-empty".into());
        }
        if self
            .evidence
            .artifact_checksums
            .iter()
            .any(|(path, checksum)| {
                checksum.trim().is_empty()
                    || !self.artifacts.iter().any(|artifact| artifact == path)
            })
        {
            return Err(
                "QA artifact checksums must be non-empty and refer to declared artifacts".into(),
            );
        }
        if self
            .evidence
            .artifact_provenance
            .iter()
            .any(|(path, provenance)| {
                !self.artifacts.iter().any(|artifact| artifact == path)
                    || provenance.source.trim().is_empty()
                    || provenance.generation.trim().is_empty()
                    || provenance.checksum.trim().is_empty()
                    || provenance.version.trim().is_empty()
                    || self.evidence.artifact_checksums.get(path) != Some(&provenance.checksum)
            })
        {
            return Err(
                "QA artifact provenance must name declared artifacts with matching checksums"
                    .into(),
            );
        }
        if self
            .metrics
            .values()
            .chain(self.thresholds.values())
            .any(|value| !value.is_finite())
        {
            return Err("QA result metrics and thresholds must be finite".into());
        }
        if self
            .metrics
            .keys()
            .chain(self.thresholds.keys())
            .chain(self.threshold_directions.keys())
            .chain(self.environment.keys())
            .any(|name| name.trim().is_empty())
        {
            return Err(
                "QA result metric, threshold, and environment names must be non-empty".into(),
            );
        }
        if self
            .threshold_directions
            .keys()
            .any(|name| !self.thresholds.contains_key(name))
        {
            return Err("QA threshold direction has no numeric threshold".into());
        }
        let mut artifact_paths = std::collections::BTreeSet::new();
        if self.artifacts.iter().any(|artifact| {
            !artifact_paths.insert(artifact)
                || artifact.trim().is_empty()
                || artifact.contains('\\')
                || artifact.contains(':')
                || Path::new(artifact).is_absolute()
                || Path::new(artifact).components().any(|component| {
                    matches!(
                        component,
                        std::path::Component::ParentDir | std::path::Component::CurDir
                    )
                })
        }) {
            return Err(
                "QA result artifact paths must be non-empty, forward-slash relative paths".into(),
            );
        }
        Ok(())
    }

    /// Validate that this record is eligible for a release-critical gate.
    ///
    /// A syntactically valid skipped record is useful for diagnostics, but it
    /// must never be promoted to a passing release result.  Likewise, when a
    /// caller requests a concrete accelerator, a passing record must identify
    /// the same backend rather than silently reporting a host fallback.
    pub fn release_eligible(&self) -> Result<(), String> {
        self.release_eligible_for(None, false, false, false, false)
    }

    /// Validate the stricter policy required by a release accelerator shard.
    ///
    /// The ordinary [`Self::release_eligible`] method remains compatible with
    /// older CPU records.  Accelerator and native-feature lanes must opt into
    /// this policy so a valid-but-incomplete legacy record cannot promote a
    /// fallback, a skipped case, or evidence with no provenance.
    pub fn release_eligible_for(
        &self,
        expected_backend: Option<&str>,
        require_thresholds: bool,
        require_runtime: bool,
        require_hardware: bool,
        require_artifact_provenance: bool,
    ) -> Result<(), String> {
        self.validate()?;
        if self.status != QaStatus::Passed {
            return Err(format!(
                "QA result status {:?} is not release-eligible",
                self.status
            ));
        }
        if require_thresholds && self.thresholds.is_empty() {
            return Err("release case must declare at least one threshold".into());
        }
        if require_runtime && (self.runtime.counts.executed == 0 || self.runtime.counts.passed == 0)
        {
            return Err("release case is missing executed/pass runtime counts".into());
        }
        if require_artifact_provenance {
            if self.artifacts.is_empty() {
                return Err("release case is missing declared artifacts".into());
            }
            if self.artifacts.iter().any(|path| {
                !self.evidence.artifact_provenance.contains_key(path)
                    || !self.evidence.artifact_checksums.contains_key(path)
            }) {
                return Err("release case has an artifact without complete provenance".into());
            }
        }
        if require_hardware {
            let hardware = &self.runtime.hardware;
            let backend = expected_backend
                .or(self.actual_backend.as_deref())
                .unwrap_or_default();
            let required = if backend.contains("cuda") || backend.contains("nvidia") {
                [
                    ("device", hardware.device.as_ref()),
                    ("driver", hardware.driver.as_ref()),
                    ("toolkit", hardware.toolkit.as_ref()),
                ]
                .into_iter()
                .collect::<Vec<_>>()
            } else if backend.contains("x86") {
                [
                    ("cpu_model", hardware.cpu_model.as_ref()),
                    ("microcode", hardware.microcode.as_ref()),
                    ("os", hardware.os.as_ref()),
                    ("compiler", hardware.compiler.as_ref()),
                    ("target_features", hardware.target_features.as_ref()),
                    ("runtime_dispatch", hardware.runtime_dispatch.as_ref()),
                    ("thread_policy", hardware.thread_policy.as_ref()),
                    ("affinity_policy", hardware.affinity_policy.as_ref()),
                    ("numa_policy", hardware.numa_policy.as_ref()),
                    ("deterministic_mode", hardware.deterministic_mode.as_ref()),
                ]
                .into_iter()
                .collect::<Vec<_>>()
            } else {
                [
                    ("cpu_model", hardware.cpu_model.as_ref()),
                    ("os", hardware.os.as_ref()),
                    ("compiler", hardware.compiler.as_ref()),
                ]
                .into_iter()
                .collect::<Vec<_>>()
            };
            if let Some((name, _)) = required
                .into_iter()
                .find(|(_, value)| !usable_provenance(*value))
            {
                return Err(format!(
                    "release case is missing hardware provenance {name:?}"
                ));
            }
        }
        if let Some(requested) = self.requested_backend.as_deref() {
            let actual = self.actual_backend.as_deref().ok_or_else(|| {
                format!(
                    "passing QA result requested backend {requested:?} but did not report actual_backend"
                )
            })?;
            if actual != requested {
                return Err(format!(
                    "passing QA result used backend {actual:?}, requested {requested:?}"
                ));
            }
        }
        if let Some(expected) = expected_backend {
            if self.requested_backend.as_deref() != Some(expected)
                || self.actual_backend.as_deref() != Some(expected)
            {
                return Err(format!(
                    "release case must report requested and actual backend {expected:?}"
                ));
            }
            if !self
                .backend_outcomes
                .iter()
                .any(|outcome| outcome.backend == expected && outcome.status == QaStatus::Passed)
            {
                return Err(format!(
                    "release case did not record a passing outcome for backend {expected:?}"
                ));
            }
        }
        for (name, &threshold) in &self.thresholds {
            let direction = self.threshold_directions.get(name).ok_or_else(|| {
                format!("release threshold {name:?} is missing an explicit direction")
            })?;
            let metric = self
                .metrics
                .get(name)
                .ok_or_else(|| format!("release threshold {name:?} has no recorded metric"))?;
            let satisfied = match direction {
                QaThresholdDirection::Minimum => *metric >= threshold,
                QaThresholdDirection::Maximum => *metric <= threshold,
            };
            if !satisfied {
                return Err(format!(
                    "release threshold {name:?} is not satisfied: metric={metric}, threshold={threshold}, direction={direction:?}"
                ));
            }
        }
        Ok(())
    }

    pub fn to_json_line(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| error.to_string())
    }

    /// Parse and validate one machine-readable result record. QA runners may
    /// prefix records with `QA_RESULT:` while sharing stdout with legacy
    /// human markers; the prefix is intentionally accepted here.
    pub fn from_json_line(line: &str) -> Result<Self, String> {
        let trimmed = line.trim();
        let payload = trimmed
            .strip_prefix("QA_RESULT:")
            .map(str::trim)
            .unwrap_or(trimmed);
        if payload.is_empty() {
            return Err("QA result line is empty".into());
        }
        let result: Self = serde_json::from_str(payload)
            .map_err(|error| format!("invalid QA result JSON: {error}"))?;
        result.validate()?;
        Ok(result)
    }

    /// Read and validate a pretty-printed JSON result artifact.
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .map_err(|error| format!("read QA result {}: {error}", path.display()))?;
        Self::from_json_line(&contents)
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let bytes = serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?;
        std::fs::write(path, bytes).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passed_result_serializes_metrics_and_provenance() {
        let result = QaResult::passed("compute.properties.v1")
            .metric("relative_error", 1.0e-8)
            .threshold("relative_error", 1.0e-6)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .backend_passed("cpu")
            .seed("seed=7")
            .artifact_with_checksum("result.json", "sha256:abc")
            .input_checksum("sha256:input")
            .reference("independent scalar reference")
            .reference_checksum("sha256:reference")
            .dimension("dense 2x3")
            .formulation("GEMV")
            .tolerance("relative <= 1e-6")
            .memory_bytes(4096)
            .timing_ms("kernel", 7)
            .elapsed_ms(12);
        let line = result.to_json_line().unwrap();
        let decoded: QaResult = serde_json::from_str(&line).unwrap();
        assert_eq!(decoded.status, QaStatus::Passed);
        assert_eq!(decoded.backend_outcomes[0].backend, "cpu");
        assert_eq!(decoded.metrics["relative_error"], 1.0e-8);
        assert_eq!(decoded.seed.as_deref(), Some("seed=7"));
        assert_eq!(
            decoded.evidence.artifact_checksums["result.json"],
            "sha256:abc"
        );
        assert_eq!(decoded.evidence.memory_bytes, Some(4096));
        assert_eq!(decoded.evidence.timings_ms["kernel"], 7);
    }

    #[test]
    fn evidence_rejects_blank_values_and_undeclared_artifact_checksums() {
        assert!(QaResult::passed("case").reference(" ").validate().is_err());
        assert!(
            QaResult::passed("case")
                .timing_ms(" ", 1)
                .validate()
                .is_err()
        );

        let mut result = QaResult::passed("case");
        result
            .evidence
            .artifact_checksums
            .insert("missing.json".into(), "sha256:abc".into());
        assert!(result.validate().is_err());
    }

    #[test]
    fn skipped_result_requires_a_reason() {
        let mut result = QaResult::passed("case");
        result.status = QaStatus::Skipped;
        assert!(result.validate().is_err());
        assert!(
            QaResult::skipped("case", "accelerator unavailable")
                .to_json_line()
                .is_ok()
        );
    }

    #[test]
    fn backend_outcomes_require_typed_reasons_and_unique_names() {
        let passing = QaResult::passed("case")
            .backend_passed("cpu")
            .backend_skipped("metal", "Metal feature is unavailable");
        let decoded: QaResult = serde_json::from_str(&passing.to_json_line().unwrap()).unwrap();
        assert_eq!(decoded.backend_outcomes.len(), 2);
        assert!(
            QaResult::passed("case")
                .backend_skipped("metal", " ")
                .validate()
                .is_err()
        );
        assert!(
            QaResult::passed("case")
                .backend_passed("cpu")
                .backend_passed("cpu")
                .validate()
                .is_err()
        );
    }

    #[test]
    fn non_skipped_results_cannot_hide_a_skip_reason() {
        let mut passed = QaResult::passed("case");
        passed.skip_reason = Some("backend unavailable".into());
        assert!(passed.validate().is_err());
        assert!(passed.release_eligible().is_err());

        let mut failed = QaResult::failed("case", "assertion failed");
        failed.skip_reason = Some("backend unavailable".into());
        assert!(failed.validate().is_err());

        assert!(
            QaResult::passed("case")
                .requested_backend("   ")
                .validate()
                .is_err()
        );
    }

    #[test]
    fn failed_results_and_optional_seeds_require_nonblank_provenance() {
        assert!(QaResult::failed("case", " ").validate().is_err());
        assert!(QaResult::passed("case").seed("\t").validate().is_err());

        let mut raw_failed = QaResult::passed("case");
        raw_failed.status = QaStatus::Failed;
        let raw = serde_json::to_string(&raw_failed).expect("raw test JSON");
        assert!(QaResult::from_json_line(&raw).is_err());
    }

    #[test]
    fn result_validation_rejects_blank_structured_field_names() {
        let mut metric = QaResult::passed("case");
        metric.metrics.insert(" ".into(), 1.0);
        assert!(metric.validate().is_err());

        let mut threshold = QaResult::passed("case");
        threshold.thresholds.insert("".into(), 1.0);
        assert!(threshold.validate().is_err());

        let mut direction = QaResult::passed("case");
        direction
            .threshold_directions
            .insert("\t".into(), QaThresholdDirection::Maximum);
        assert!(direction.validate().is_err());

        let mut environment = QaResult::passed("case");
        environment.environment.insert("\n".into(), "value".into());
        assert!(environment.validate().is_err());
    }

    #[test]
    fn nonfinite_metrics_are_rejected() {
        let result = QaResult::passed("case").metric("residual", f64::NAN);
        assert!(result.to_json_line().is_err());
    }

    #[test]
    fn parser_accepts_runner_prefix_and_rejects_unsafe_artifacts() {
        let line = QaResult::passed("case")
            .artifact("results/result.json")
            .to_json_line()
            .unwrap();
        let decoded = QaResult::from_json_line(&format!("QA_RESULT: {line}")).unwrap();
        assert_eq!(decoded.case_id, "case");

        let unsafe_result = QaResult::passed("case").artifact("../outside.json");
        assert!(unsafe_result.to_json_line().is_err());
        assert!(
            QaResult::passed("case")
                .artifact("..\\outside.json")
                .to_json_line()
                .is_err()
        );
        assert!(
            QaResult::passed("case")
                .artifact("C:\\release\\result.json")
                .to_json_line()
                .is_err()
        );
        assert!(
            QaResult::passed("case")
                .artifact("./result.json")
                .to_json_line()
                .is_err()
        );
        assert!(
            QaResult::passed("case")
                .artifact("result.json")
                .artifact("result.json")
                .to_json_line()
                .is_err()
        );
        assert!(QaResult::from_json_line("QA_RESULT: not-json").is_err());
    }

    #[test]
    fn release_eligibility_rejects_skips_and_backend_fallbacks() {
        assert!(
            QaResult::skipped("case", "accelerator unavailable")
                .release_eligible()
                .is_err()
        );
        let fallback = QaResult::passed("case")
            .requested_backend("cuda")
            .actual_backend("cpu");
        assert!(fallback.release_eligible().is_err());
        let passing = QaResult::passed("case")
            .requested_backend("cpu")
            .actual_backend("cpu");
        assert!(passing.release_eligible().is_ok());
    }

    #[test]
    fn release_eligibility_evaluates_explicit_threshold_directions() {
        let passing = QaResult::passed("case")
            .metric("max_error", 0.25)
            .maximum("max_error", 0.5)
            .metric("min_rate", 2.0)
            .minimum("min_rate", 1.5);
        assert!(passing.release_eligible().is_ok());

        let failing = QaResult::passed("case")
            .metric("max_error", 0.75)
            .maximum("max_error", 0.5);
        assert!(failing.release_eligible().is_err());
    }

    #[test]
    fn release_eligibility_rejects_ambiguous_or_missing_threshold_metrics() {
        let ambiguous = QaResult::passed("case")
            .metric("error", 0.1)
            .threshold("error", 1.0);
        assert!(ambiguous.release_eligible().is_err());

        let missing_metric = QaResult::passed("case").maximum("error", 1.0);
        assert!(missing_metric.release_eligible().is_err());
    }
}
