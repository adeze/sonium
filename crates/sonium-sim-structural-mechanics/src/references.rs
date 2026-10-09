//! Versioned published-reference data and comparison contracts.
//!
//! The crate does not embed or invent official NAFEMS values. A caller must
//! provide a manifest containing citation and applicability information for
//! the reference data it wants to compare against. Loading and checking that
//! manifest is deliberately separate from the analytical validation catalogue.

use std::{collections::HashSet, fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    NafemsBenchmarkKind, NafemsBenchmarkSuite, STRUCTURAL_VALIDATION_SCHEMA_VERSION,
    StructuralMechanicsError,
};

/// Schema for the JSON manifest accepted by [`load_published_reference_manifest`].
pub const PUBLISHED_REFERENCE_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const PUBLISHED_REFERENCE_MANIFEST_DIGEST_ALGORITHM: &str = "sha256";
/// Schema for [`PublishedReferenceComparisonReport`].
pub const PUBLISHED_REFERENCE_COMPARISON_SCHEMA_VERSION: u32 = 1;

/// Declared origin of values in a published-reference manifest.
///
/// This is provenance supplied by the manifest author. It is not an
/// authenticity check and is therefore reported alongside the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublishedReferenceAuthority {
    Nafems,
    PublishedLiterature,
    ExternalSolver,
    UserSupplied,
}

impl PublishedReferenceAuthority {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nafems => "NAFEMS",
            Self::PublishedLiterature => "published literature",
            Self::ExternalSolver => "external solver",
            Self::UserSupplied => "user supplied",
        }
    }
}

/// One reference value and the native validation case it describes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublishedReferenceCase {
    /// Stable identifier matching a case in [`NafemsBenchmarkSuite`].
    pub id: String,
    pub kind: NafemsBenchmarkKind,
    pub quantity: String,
    pub units: String,
    pub reference_value: f64,
    pub tolerance: f64,
    /// Applicability of this value, such as mesh/order or boundary assumptions.
    pub applicability: String,
}

/// Versioned, externally supplied reference data.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PublishedReferenceManifest {
    pub schema_version: u32,
    pub manifest_id: String,
    pub title: String,
    pub authority: PublishedReferenceAuthority,
    pub citation: String,
    pub source_uri: Option<String>,
    pub source_revision: String,
    pub cases: Vec<PublishedReferenceCase>,
    pub applicability_limits: Vec<String>,
    /// Optional digest binding all manifest metadata and reference values.
    /// It protects input integrity but does not authenticate the cited source.
    #[serde(default)]
    pub manifest_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PublishedReferenceManifestWire {
    schema_version: u32,
    manifest_id: String,
    title: String,
    authority: PublishedReferenceAuthority,
    citation: String,
    source_uri: Option<String>,
    source_revision: String,
    cases: Vec<PublishedReferenceCase>,
    applicability_limits: Vec<String>,
    #[serde(default)]
    manifest_sha256: Option<String>,
}

impl<'de> Deserialize<'de> for PublishedReferenceManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PublishedReferenceManifestWire::deserialize(deserializer)?;
        let manifest = Self {
            schema_version: wire.schema_version,
            manifest_id: wire.manifest_id,
            title: wire.title,
            authority: wire.authority,
            citation: wire.citation,
            source_uri: wire.source_uri,
            source_revision: wire.source_revision,
            cases: wire.cases,
            applicability_limits: wire.applicability_limits,
            manifest_sha256: wire.manifest_sha256,
        };
        manifest
            .validate()
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))?;
        Ok(manifest)
    }
}

impl PublishedReferenceManifest {
    /// Validate schema, provenance fields, numerical values, and case keys.
    pub fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.schema_version != PUBLISHED_REFERENCE_MANIFEST_SCHEMA_VERSION {
            return Err(manifest_invalid(
                "schema_version",
                self.schema_version.to_string(),
            ));
        }
        for (field, value) in [
            ("manifest_id", self.manifest_id.as_str()),
            ("title", self.title.as_str()),
            ("citation", self.citation.as_str()),
            ("source_revision", self.source_revision.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(manifest_invalid(field, "empty"));
            }
        }
        if self
            .source_uri
            .as_deref()
            .is_some_and(|uri| uri.trim().is_empty())
        {
            return Err(manifest_invalid("source_uri", "empty"));
        }
        if self.cases.is_empty() {
            return Err(manifest_invalid("cases", "empty"));
        }

        let mut ids = HashSet::with_capacity(self.cases.len());
        for case in &self.cases {
            if case.id.trim().is_empty() {
                return Err(manifest_invalid("case.id", "empty"));
            }
            if !ids.insert(case.id.as_str()) {
                return Err(manifest_invalid(
                    "case.id",
                    format!("duplicate `{}`", case.id),
                ));
            }
            for (field, value) in [
                ("case.quantity", case.quantity.as_str()),
                ("case.units", case.units.as_str()),
                ("case.applicability", case.applicability.as_str()),
            ] {
                if value.trim().is_empty() {
                    return Err(manifest_invalid(field, "empty"));
                }
            }
            if !case.reference_value.is_finite() {
                return Err(manifest_invalid(
                    "case.reference_value",
                    case.reference_value.to_string(),
                ));
            }
            if !case.tolerance.is_finite() || case.tolerance < 0.0 {
                return Err(manifest_invalid(
                    "case.tolerance",
                    case.tolerance.to_string(),
                ));
            }
        }
        if let Some(manifest_sha256) = &self.manifest_sha256 {
            let expected = published_reference_manifest_sha256(self);
            if manifest_sha256.len() != 64
                || !manifest_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                || !manifest_sha256.eq_ignore_ascii_case(&expected)
            {
                return Err(manifest_invalid(
                    "manifest_sha256",
                    "digest does not match manifest contents",
                ));
            }
        }
        Ok(())
    }

    /// Load and validate a manifest from UTF-8 JSON.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, StructuralMechanicsError> {
        load_published_reference_manifest(path)
    }
}

/// Load and validate a versioned published-reference JSON manifest.
pub fn load_published_reference_manifest(
    path: impl AsRef<Path>,
) -> Result<PublishedReferenceManifest, StructuralMechanicsError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path).map_err(|error| {
        StructuralMechanicsError::PublishedReferenceIo {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    let manifest =
        serde_json::from_str::<PublishedReferenceManifest>(&contents).map_err(|error| {
            StructuralMechanicsError::PublishedReferenceIo {
                path: path.to_path_buf(),
                message: format!("invalid JSON: {error}"),
            }
        })?;
    manifest.validate()?;
    Ok(manifest)
}

/// One case-level comparison against a published reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublishedReferenceComparisonCase {
    pub id: String,
    pub kind: NafemsBenchmarkKind,
    pub quantity: String,
    pub units: String,
    pub computed_value: Option<f64>,
    pub reference_value: f64,
    pub relative_error: Option<f64>,
    pub tolerance: f64,
    pub kind_matches: bool,
    pub solver_gate_passed: bool,
    pub passed: bool,
    pub applicability: String,
}

/// Comparison report with explicit provenance and coverage information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublishedReferenceComparisonReport {
    pub schema_version: u32,
    pub manifest_schema_version: u32,
    pub manifest_id: String,
    pub authority: PublishedReferenceAuthority,
    pub citation: String,
    pub source_uri: Option<String>,
    pub source_revision: String,
    pub cases: Vec<PublishedReferenceComparisonCase>,
    /// Native suite cases not represented by this manifest.
    pub unmatched_suite_case_ids: Vec<String>,
    pub applicability_limits: Vec<String>,
    pub all_passed: bool,
    /// True means the manifest passed structural validation; it does not mean
    /// the source was independently authenticated or that values are official.
    pub reference_manifest_validated: bool,
    pub provenance_verified: bool,
    #[serde(default)]
    pub manifest_integrity: PublishedReferenceManifestIntegrity,
}

/// Integrity state for a caller-supplied published-reference manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PublishedReferenceManifestIntegrity {
    #[default]
    Unbound,
    Sha256Validated,
}

impl PublishedReferenceComparisonReport {
    /// Return the gate that is safe to use as published-reference parity.
    ///
    /// Numerical agreement and schema validation alone do not establish that
    /// the cited source was independently verified. Callers must set
    /// `provenance_verified` only after checking the source and revision
    /// recorded in the report.
    pub fn parity_passed(&self) -> bool {
        self.all_passed && self.reference_manifest_validated && self.provenance_verified
    }
}

/// Compare a native validation suite with caller-provided reference values.
pub fn compare_nafems_suite_to_published_manifest(
    suite: &NafemsBenchmarkSuite,
    manifest: &PublishedReferenceManifest,
) -> Result<PublishedReferenceComparisonReport, StructuralMechanicsError> {
    manifest.validate()?;

    if suite.schema_version != STRUCTURAL_VALIDATION_SCHEMA_VERSION {
        return Err(manifest_invalid(
            "suite.schema_version",
            suite.schema_version.to_string(),
        ));
    }

    let mut cases = Vec::with_capacity(manifest.cases.len());
    for reference in &manifest.cases {
        let native = suite.cases.iter().find(|case| case.id == reference.id);
        let (computed_value, relative_error, kind_matches, solver_gate_passed) =
            if let Some(native) = native {
                let kind_matches = native.kind == reference.kind;
                let computed_value = native
                    .computed_value
                    .is_finite()
                    .then_some(native.computed_value);
                let relative_error = computed_value.map(|value| {
                    (value - reference.reference_value).abs()
                        / reference.reference_value.abs().max(1.0e-30)
                });
                (computed_value, relative_error, kind_matches, native.passed)
            } else {
                (None, None, false, false)
            };
        let passed = kind_matches
            && solver_gate_passed
            && relative_error.is_some_and(|error| error <= reference.tolerance);
        cases.push(PublishedReferenceComparisonCase {
            id: reference.id.clone(),
            kind: reference.kind,
            quantity: reference.quantity.clone(),
            units: reference.units.clone(),
            computed_value,
            reference_value: reference.reference_value,
            relative_error,
            tolerance: reference.tolerance,
            kind_matches,
            solver_gate_passed,
            passed,
            applicability: reference.applicability.clone(),
        });
    }

    let unmatched_suite_case_ids = suite
        .cases
        .iter()
        .filter(|native| {
            !manifest
                .cases
                .iter()
                .any(|reference| reference.id == native.id)
        })
        .map(|native| native.id.clone())
        .collect::<Vec<_>>();
    let all_passed = !cases.is_empty()
        && unmatched_suite_case_ids.is_empty()
        && cases.iter().all(|case| case.passed);

    Ok(PublishedReferenceComparisonReport {
        schema_version: PUBLISHED_REFERENCE_COMPARISON_SCHEMA_VERSION,
        manifest_schema_version: manifest.schema_version,
        manifest_id: manifest.manifest_id.clone(),
        authority: manifest.authority,
        citation: manifest.citation.clone(),
        source_uri: manifest.source_uri.clone(),
        source_revision: manifest.source_revision.clone(),
        cases,
        unmatched_suite_case_ids,
        applicability_limits: manifest.applicability_limits.clone(),
        all_passed,
        reference_manifest_validated: true,
        provenance_verified: false,
        manifest_integrity: if manifest.manifest_sha256.is_some() {
            PublishedReferenceManifestIntegrity::Sha256Validated
        } else {
            PublishedReferenceManifestIntegrity::Unbound
        },
    })
}

/// Compute the canonical SHA-256 binding for a published-reference manifest.
///
/// This binds the metadata and values consumed by comparison. It is an
/// integrity check, not a signature or independent verification of the cited
/// NAFEMS/literature/external source.
pub fn published_reference_manifest_sha256(manifest: &PublishedReferenceManifest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PUBLISHED_REFERENCE_MANIFEST_SCHEMA_VERSION.to_le_bytes());
    update_digest_string(&mut hasher, &manifest.manifest_id);
    update_digest_string(&mut hasher, &manifest.title);
    update_digest_string(&mut hasher, manifest.authority.as_str());
    update_digest_string(&mut hasher, &manifest.citation);
    match &manifest.source_uri {
        Some(uri) => {
            hasher.update([1_u8]);
            update_digest_string(&mut hasher, uri);
        }
        None => hasher.update([0_u8]),
    }
    update_digest_string(&mut hasher, &manifest.source_revision);
    hasher.update((manifest.cases.len() as u64).to_le_bytes());
    for case in &manifest.cases {
        update_digest_string(&mut hasher, &case.id);
        update_digest_string(&mut hasher, benchmark_kind_tag(case.kind));
        update_digest_string(&mut hasher, &case.quantity);
        update_digest_string(&mut hasher, &case.units);
        hasher.update(case.reference_value.to_bits().to_le_bytes());
        hasher.update(case.tolerance.to_bits().to_le_bytes());
        update_digest_string(&mut hasher, &case.applicability);
    }
    hasher.update((manifest.applicability_limits.len() as u64).to_le_bytes());
    for limit in &manifest.applicability_limits {
        update_digest_string(&mut hasher, limit);
    }
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn benchmark_kind_tag(kind: NafemsBenchmarkKind) -> &'static str {
    match kind {
        NafemsBenchmarkKind::Static => "static",
        NafemsBenchmarkKind::Modal => "modal",
        NafemsBenchmarkKind::ThermalStress => "thermal-stress",
        NafemsBenchmarkKind::Nonlinear => "nonlinear",
    }
}

fn update_digest_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn manifest_invalid(field: &'static str, value: impl Into<String>) -> StructuralMechanicsError {
    StructuralMechanicsError::PublishedReferenceManifestInvalid {
        field,
        value: value.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn manifest_for_suite(suite: &NafemsBenchmarkSuite) -> PublishedReferenceManifest {
        PublishedReferenceManifest {
            schema_version: PUBLISHED_REFERENCE_MANIFEST_SCHEMA_VERSION,
            manifest_id: "test-reference-manifest".to_string(),
            title: "Test reference values".to_string(),
            authority: PublishedReferenceAuthority::UserSupplied,
            citation: "Test fixture; not a published result".to_string(),
            source_uri: Some("https://example.invalid/reference.json".to_string()),
            source_revision: "fixture-1".to_string(),
            cases: suite
                .cases
                .iter()
                .map(|case| PublishedReferenceCase {
                    id: case.id.clone(),
                    kind: case.kind,
                    quantity: "validation quantity".to_string(),
                    units: "SI".to_string(),
                    reference_value: case.computed_value,
                    tolerance: 1.0e-12,
                    applicability: "matches the native analytical fixture".to_string(),
                })
                .collect(),
            applicability_limits: vec!["Test fixture only".to_string()],
            manifest_sha256: None,
        }
    }

    #[test]
    fn manifest_round_trip_loads_and_comparison_reports_provenance() {
        let suite = crate::run_nafems_benchmark_suite().unwrap();
        let manifest = manifest_for_suite(&suite);
        let path = std::env::temp_dir().join(format!(
            "sonium-published-reference-{}-{}.json",
            std::process::id(),
            PUBLISHED_REFERENCE_MANIFEST_SCHEMA_VERSION
        ));
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let loaded = load_published_reference_manifest(&path).unwrap();
        let _ = fs::remove_file(&path);

        let report = compare_nafems_suite_to_published_manifest(&suite, &loaded).unwrap();
        assert!(report.all_passed, "{report:#?}");
        assert!(report.reference_manifest_validated);
        assert!(!report.provenance_verified);
        assert!(!report.parity_passed());
        assert!(report.unmatched_suite_case_ids.is_empty());

        let mut verified_report = report.clone();
        verified_report.provenance_verified = true;
        assert!(verified_report.parity_passed());
    }

    #[test]
    fn manifest_digest_round_trip_and_tamper_gate_hold() {
        let suite = crate::run_nafems_benchmark_suite().unwrap();
        let mut manifest = manifest_for_suite(&suite);
        let digest = published_reference_manifest_sha256(&manifest);
        manifest.manifest_sha256 = Some(digest.to_uppercase());

        let encoded = serde_json::to_vec(&manifest).unwrap();
        let decoded: PublishedReferenceManifest = serde_json::from_slice(&encoded).unwrap();
        assert!(
            decoded
                .manifest_sha256
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case(&digest))
        );
        let report = compare_nafems_suite_to_published_manifest(&suite, &decoded).unwrap();
        assert_eq!(
            report.manifest_integrity,
            PublishedReferenceManifestIntegrity::Sha256Validated
        );
        assert!(!report.provenance_verified);
        assert!(!report.parity_passed());

        let mut tampered = decoded;
        tampered.cases[0].reference_value += 1.0;
        let error = tampered
            .validate()
            .expect_err("tampered manifest must fail closed");
        assert!(error.to_string().contains("manifest_sha256"));
    }

    #[test]
    fn manifest_digest_changes_when_applicability_changes() {
        let suite = crate::run_nafems_benchmark_suite().unwrap();
        let manifest = manifest_for_suite(&suite);
        let original = published_reference_manifest_sha256(&manifest);
        let mut changed = manifest.clone();
        changed
            .applicability_limits
            .push("changed fixture bound".to_string());
        assert_ne!(original, published_reference_manifest_sha256(&changed));
    }

    #[test]
    fn comparison_requires_complete_native_case_coverage() {
        let suite = crate::run_nafems_benchmark_suite().unwrap();
        let mut manifest = manifest_for_suite(&suite);
        manifest.cases.pop();

        let report = compare_nafems_suite_to_published_manifest(&suite, &manifest).unwrap();
        assert!(!report.all_passed);
        assert_eq!(report.unmatched_suite_case_ids.len(), 1);
        assert!(!report.parity_passed());
    }

    #[test]
    fn comparison_keeps_missing_cases_as_failed_coverage() {
        let suite = crate::run_nafems_benchmark_suite().unwrap();
        let mut manifest = manifest_for_suite(&suite);
        manifest.cases[0].id = "not-in-native-suite".to_string();
        let report = compare_nafems_suite_to_published_manifest(&suite, &manifest).unwrap();
        assert!(!report.all_passed);
        assert!(!report.cases[0].passed);
        assert!(report.cases[0].computed_value.is_none());
        assert!(!report.cases[0].kind_matches);
        assert!(!report.cases[0].solver_gate_passed);
    }

    #[test]
    fn manifest_validation_rejects_schema_duplicates_and_negative_tolerance() {
        let suite = crate::run_nafems_benchmark_suite().unwrap();
        let mut manifest = manifest_for_suite(&suite);
        manifest.schema_version += 1;
        assert!(matches!(
            manifest.validate(),
            Err(
                StructuralMechanicsError::PublishedReferenceManifestInvalid {
                    field: "schema_version",
                    ..
                }
            )
        ));

        let mut manifest = manifest_for_suite(&suite);
        manifest.cases[0].tolerance = -1.0;
        assert!(matches!(
            manifest.validate(),
            Err(
                StructuralMechanicsError::PublishedReferenceManifestInvalid {
                    field: "case.tolerance",
                    ..
                }
            )
        ));

        let mut manifest = manifest_for_suite(&suite);
        manifest.cases.push(manifest.cases[0].clone());
        assert!(matches!(
            manifest.validate(),
            Err(
                StructuralMechanicsError::PublishedReferenceManifestInvalid {
                    field: "case.id",
                    ..
                }
            )
        ));
    }

    #[test]
    fn manifest_serde_admission_rejects_invalid_reference_artifacts() {
        let suite = crate::run_nafems_benchmark_suite().unwrap();
        let manifest = manifest_for_suite(&suite);
        let encoded = serde_json::to_value(&manifest).unwrap();
        let decoded: PublishedReferenceManifest = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, manifest);

        let mut malformed = encoded;
        malformed["cases"][0]["tolerance"] = serde_json::json!(-1.0);
        let error = serde_json::from_value::<PublishedReferenceManifest>(malformed)
            .expect_err("invalid reference JSON must be rejected at deserialization");
        assert!(error.to_string().contains("case.tolerance"));
    }

    #[test]
    fn manifest_loader_reports_io_and_json_errors_as_typed_failures() {
        let missing = std::env::temp_dir().join(format!(
            "sonium-published-reference-missing-{}",
            std::process::id()
        ));
        assert!(matches!(
            load_published_reference_manifest(&missing),
            Err(StructuralMechanicsError::PublishedReferenceIo { .. })
        ));

        let invalid = std::env::temp_dir().join(format!(
            "sonium-published-reference-invalid-{}.json",
            std::process::id()
        ));
        fs::write(&invalid, b"not json").unwrap();
        assert!(matches!(
            PublishedReferenceManifest::load_from_path(&invalid),
            Err(StructuralMechanicsError::PublishedReferenceIo { .. })
        ));
        let _ = fs::remove_file(invalid);
    }
}
