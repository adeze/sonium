//! Long-lived compiler facade and canonical-IR preview APIs.

use crate::imports::ImportResolver;
use crate::{
    analyze_flattened_ast, build_artifact_sections, build_result_with_cached_sections,
    flatten_imports_with_resolver, preview_provenance_manifest_value, CachedArtifactSections,
    CanonicalIrCache, CompileError, CompileOptions, CompileResult, DiagLevel, Diagnostic,
    FrontendProduct, GeoCodegen, PythonCodegen,
};
use ed25519_zebra::{Signature, VerificationKey, VerificationKeyBytes};
use meshlang_ast::{
    span::Span, CanonicalGeometryIr, Declaration, SourceFile, StructuralHash, VersionedCanonicalIr,
    CURRENT_IR_DOCUMENT_VERSION,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Observable state for a long-lived [`IncrementalCompiler`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct IncrementalCompilerStats {
    pub parsed_module_entries: usize,
    pub parsed_module_hits: u64,
    pub parsed_module_misses: u64,
    pub canonical_ir_entries: usize,
    pub canonical_ir_hits: u64,
    pub canonical_ir_misses: u64,
    pub backend_artifact_entries: usize,
    pub backend_artifact_hits: u64,
    pub backend_artifact_misses: u64,
    pub artifact_section_entries: usize,
    pub artifact_section_hits: u64,
    pub artifact_section_misses: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BackendKind {
    Geo,
    Python,
}

/// Typed result for editor, optimizer, and browser canonical-IR previews.
#[derive(Debug)]
pub struct CanonicalIrPreview {
    pub canonical_ir: Arc<CanonicalGeometryIr>,
    pub canonical_ir_hash: StructuralHash,
    pub diagnostics: Vec<Diagnostic>,
    pub sensitivity_manifest: Value,
    pub provenance_manifest: Value,
}
/// Versioned, host-independent wire representation of a canonical-IR preview.
///
/// The native and WASM preview entry points both emit this envelope. Keeping
/// the representation typed makes editor integrations able to validate the
/// schema and the IR identity before consuming an artifact from an untrusted
/// transport. Sensitivity and provenance manifests travel with the preview
/// so browser/editor consumers do not need a backend sidecar round trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalIrPreviewEnvelope {
    pub preview_schema: u32,
    pub compiler: String,
    pub compiler_version: String,
    pub canonical_ir_hash: String,
    pub canonical_ir: VersionedCanonicalIr,
    pub diagnostics: Vec<CanonicalIrPreviewDiagnostic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensitivity_manifest: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_manifest: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry_manifest_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry_signature_key_id: Option<String>,
}

/// Diagnostic level used by [`CanonicalIrPreviewEnvelope`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CanonicalIrPreviewDiagnosticLevel {
    Error,
    Warning,
    Info,
}

/// Serializable diagnostic in a canonical-IR preview envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalIrPreviewDiagnostic {
    pub level: CanonicalIrPreviewDiagnosticLevel,
    pub message: String,
    pub span: Span,
    pub hint: Option<String>,
}

/// Versioned source-registry request accepted by the browser/editor preview.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryPreviewModule {
    pub path: String,
    pub source: String,
}

/// A detached Ed25519 signature over a canonical source-registry manifest.
///
/// The public key is intentionally carried with the signature so a browser or
/// editor can verify an untrusted registry without filesystem or key-store
/// access. Hosts can enforce key allow-lists with [`RegistrySignaturePolicy`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryManifestSignature {
    pub algorithm: String,
    pub key_id: String,
    /// Lowercase hexadecimal Ed25519 public key (32 bytes).
    pub public_key: String,
    /// Lowercase hexadecimal Ed25519 signature (64 bytes).
    pub signature: String,
}

/// A host-supplied public-key pin for authenticated registry previews.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryTrustAnchor {
    pub key_id: String,
    /// Lowercase hexadecimal Ed25519 public key (32 bytes).
    pub public_key: String,
}

/// Explicit trust policy for signed registry previews.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct RegistrySignaturePolicy {
    pub trusted_keys: Vec<RegistryTrustAnchor>,
}

/// Deterministic trust-store update supplied by a host or package manager.
/// The compiler never fetches keys itself; callers deliver authenticated
/// additions and revocations, and this operation applies them atomically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryKeyRotation {
    #[serde(default)]
    pub additions: Vec<RegistryTrustAnchor>,
    #[serde(default)]
    pub revocations: Vec<String>,
}

/// Versioned, issuer-authenticated trust-store update envelope.
///
/// The issuer must already be pinned by the current policy. This prevents a
/// fetched rotation document from introducing its own trust root before its
/// signature is verified.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryKeyRotationEnvelope {
    pub schema_version: u32,
    pub rotation: RegistryKeyRotation,
    pub signature: RegistryManifestSignature,
}

/// Current authenticated key-rotation envelope schema.
pub const REGISTRY_KEY_ROTATION_SCHEMA_VERSION: u32 = 1;
/// Maximum serialized key-rotation envelope accepted at the compiler boundary.
pub const MAX_REGISTRY_KEY_ROTATION_BYTES: usize = 64 * 1024;

impl RegistryKeyRotationEnvelope {
    /// Build the domain-separated bytes signed by the pinned issuer.
    pub fn signing_payload(&self) -> Result<Vec<u8>, RegistrySignatureError> {
        if self.schema_version != REGISTRY_KEY_ROTATION_SCHEMA_VERSION {
            return Err(RegistrySignatureError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.signature.key_id.trim().is_empty() {
            return Err(RegistrySignatureError::EmptyKeyId);
        }
        let mut canonical_rotation = self.rotation.clone();
        canonical_rotation.revocations.sort();
        canonical_rotation.additions.sort_by(|left, right| {
            left.key_id
                .cmp(&right.key_id)
                .then_with(|| left.public_key.cmp(&right.public_key))
        });
        canonical_rotation.validate()?;
        let rotation_json = serde_json::to_vec(&canonical_rotation)
            .map_err(|error| RegistrySignatureError::ManifestHash(error.to_string()))?;
        let mut payload = format!(
            "sonium-meshlang-registry-key-rotation-v1\n{}\n{}\n",
            self.schema_version, self.signature.key_id
        )
        .into_bytes();
        payload.extend_from_slice(&rotation_json);
        Ok(payload)
    }

    /// Serialize the bounded envelope deterministically.
    pub fn to_json(&self) -> Result<String, RegistrySignatureError> {
        let json = serde_json::to_string(self)
            .map_err(|error| RegistrySignatureError::ManifestHash(error.to_string()))?;
        if json.len() > MAX_REGISTRY_KEY_ROTATION_BYTES {
            return Err(RegistrySignatureError::PayloadTooLarge {
                field: "key rotation",
                max: MAX_REGISTRY_KEY_ROTATION_BYTES,
            });
        }
        Ok(json)
    }

    /// Parse and validate an authenticated envelope from JSON.
    pub fn from_json(json: &str) -> Result<Self, RegistrySignatureError> {
        if json.len() > MAX_REGISTRY_KEY_ROTATION_BYTES {
            return Err(RegistrySignatureError::PayloadTooLarge {
                field: "key rotation",
                max: MAX_REGISTRY_KEY_ROTATION_BYTES,
            });
        }
        let envelope: Self = serde_json::from_str(json)
            .map_err(|error| RegistrySignatureError::ManifestHash(error.to_string()))?;
        envelope.signing_payload()?;
        Ok(envelope)
    }
}

impl RegistryKeyRotation {
    /// Validate additions and revocations without changing a policy.
    pub fn validate(&self) -> Result<(), RegistrySignatureError> {
        let mut revoked = std::collections::BTreeSet::new();
        for key_id in &self.revocations {
            if key_id.trim().is_empty() {
                return Err(RegistrySignatureError::EmptyKeyId);
            }
            if !revoked.insert(key_id.as_str()) {
                return Err(RegistrySignatureError::DuplicateKeyId(key_id.clone()));
            }
        }
        let mut additions = std::collections::BTreeSet::new();
        for anchor in &self.additions {
            if anchor.key_id.trim().is_empty() {
                return Err(RegistrySignatureError::EmptyKeyId);
            }
            if !additions.insert(anchor.key_id.as_str()) {
                return Err(RegistrySignatureError::DuplicateKeyId(
                    anchor.key_id.clone(),
                ));
            }
            decode_hex::<32>(&anchor.public_key, "public_key")?;
        }
        Ok(())
    }
}

impl RegistrySignaturePolicy {
    /// Validate anchor identities and key encodings before a preview starts.
    pub fn validate(&self) -> Result<(), RegistrySignatureError> {
        let mut key_ids = std::collections::BTreeSet::new();
        for anchor in &self.trusted_keys {
            if anchor.key_id.trim().is_empty() {
                return Err(RegistrySignatureError::EmptyKeyId);
            }
            if !key_ids.insert(anchor.key_id.as_str()) {
                return Err(RegistrySignatureError::DuplicateKeyId(
                    anchor.key_id.clone(),
                ));
            }
            decode_hex::<32>(&anchor.public_key, "public_key")?;
        }
        Ok(())
    }

    /// Apply a checked key rotation without exposing a partial policy.
    pub fn rotate(&self, rotation: &RegistryKeyRotation) -> Result<Self, RegistrySignatureError> {
        self.validate()?;
        rotation.validate()?;
        let mut revoked = std::collections::BTreeSet::new();
        for key_id in &rotation.revocations {
            revoked.insert(key_id.as_str());
            if !self
                .trusted_keys
                .iter()
                .any(|anchor| anchor.key_id == *key_id)
            {
                return Err(RegistrySignatureError::UnknownKeyId(key_id.clone()));
            }
        }

        let mut next = self
            .trusted_keys
            .iter()
            .filter(|anchor| !revoked.contains(anchor.key_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        for anchor in &rotation.additions {
            if next.iter().any(|existing| existing.key_id == anchor.key_id) {
                return Err(RegistrySignatureError::DuplicateKeyId(
                    anchor.key_id.clone(),
                ));
            }
            next.push(anchor.clone());
        }
        if next.is_empty() {
            return Err(RegistrySignatureError::EmptyTrustStore);
        }
        next.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        let policy = Self { trusted_keys: next };
        policy.validate()?;
        Ok(policy)
    }

    /// Verify a signature and require its key to be pinned by this policy.
    /// Verify an issuer-authenticated rotation and apply it atomically.
    pub fn verify_key_rotation(
        &self,
        envelope: &RegistryKeyRotationEnvelope,
    ) -> Result<Self, RegistrySignatureError> {
        if envelope.signature.algorithm != "ed25519" {
            return Err(RegistrySignatureError::UnsupportedAlgorithm);
        }
        self.validate()?;
        let anchor = self
            .trusted_keys
            .iter()
            .find(|anchor| anchor.key_id == envelope.signature.key_id)
            .ok_or_else(|| {
                RegistrySignatureError::UntrustedKeyId(envelope.signature.key_id.clone())
            })?;
        if anchor.public_key != envelope.signature.public_key {
            return Err(RegistrySignatureError::PublicKeyMismatch {
                key_id: envelope.signature.key_id.clone(),
            });
        }
        let public_key = decode_hex::<32>(&envelope.signature.public_key, "public_key")?;
        let signature_bytes = decode_hex::<64>(&envelope.signature.signature, "signature")?;
        let payload = envelope.signing_payload()?;
        let verification_key = VerificationKey::try_from(VerificationKeyBytes::from(public_key))
            .map_err(|error| RegistrySignatureError::InvalidPublicKey(format!("{error:?}")))?;
        verification_key
            .verify(&Signature::from_bytes(&signature_bytes), &payload)
            .map_err(|_| RegistrySignatureError::VerificationFailed)?;
        self.rotate(&envelope.rotation)
    }

    /// Fetch one exact rotation through a host-owned transport and apply it
    /// only after issuer authentication succeeds.
    pub fn rotate_from_transport<T: crate::registry::RegistryKeyTransport>(
        &self,
        rotation_id: &str,
        transport: &mut T,
    ) -> Result<Self, crate::registry::RegistryKeyTransportError> {
        let envelope = transport.fetch_key_rotation(rotation_id)?;
        self.verify_key_rotation(&envelope).map_err(|error| {
            crate::registry::RegistryKeyTransportError::Signature(error.to_string())
        })
    }

    pub fn verify(
        &self,
        modules: &[RegistryPreviewModule],
        signature: &RegistryManifestSignature,
    ) -> Result<StructuralHash, RegistrySignatureError> {
        self.validate()?;
        let anchor = self
            .trusted_keys
            .iter()
            .find(|anchor| anchor.key_id == signature.key_id)
            .ok_or_else(|| RegistrySignatureError::UntrustedKeyId(signature.key_id.clone()))?;
        if anchor.public_key != signature.public_key {
            return Err(RegistrySignatureError::PublicKeyMismatch {
                key_id: signature.key_id.clone(),
            });
        }
        verify_registry_manifest_signature(modules, signature)
    }

    /// Verify a signed package index against this host-owned trust policy.
    ///
    /// Package indexes carry their public key so detached envelopes remain
    /// self-contained, but that key is not trusted merely because it is in
    /// the envelope. The policy must contain the same key ID and public-key
    /// pin, keeping package-index transport and source-registry previews on
    /// the same explicit rotation boundary.
    pub fn verify_package_index(
        &self,
        index: &crate::registry::RegistryPackageIndex,
        signature: &crate::registry::RegistryPackageIndexSignature,
    ) -> Result<StructuralHash, RegistrySignatureError> {
        if signature.algorithm != "ed25519" {
            return Err(RegistrySignatureError::UnsupportedAlgorithm);
        }
        if signature.key_id.trim().is_empty() {
            return Err(RegistrySignatureError::EmptyKeyId);
        }

        self.validate()?;
        let anchor = self
            .trusted_keys
            .iter()
            .find(|anchor| anchor.key_id == signature.key_id)
            .ok_or_else(|| RegistrySignatureError::UntrustedKeyId(signature.key_id.clone()))?;
        if anchor.public_key != signature.public_key {
            return Err(RegistrySignatureError::PublicKeyMismatch {
                key_id: signature.key_id.clone(),
            });
        }

        let public_key = decode_hex::<32>(&signature.public_key, "public_key")?;
        let signature_bytes = decode_hex::<64>(&signature.signature, "signature")?;
        let index_hash = index
            .index_hash()
            .map_err(|error| RegistrySignatureError::ManifestHash(error.to_string()))?;
        let payload = index
            .signing_payload(&signature.key_id)
            .map_err(|error| RegistrySignatureError::ManifestHash(error.to_string()))?;
        let verification_key = VerificationKey::try_from(VerificationKeyBytes::from(public_key))
            .map_err(|error| RegistrySignatureError::InvalidPublicKey(format!("{error:?}")))?;
        verification_key
            .verify(&Signature::from_bytes(&signature_bytes), &payload)
            .map_err(|_| RegistrySignatureError::VerificationFailed)?;
        Ok(index_hash)
    }
}

/// Typed failures from the registry-signature boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistrySignatureError {
    #[error("registry signature algorithm must be ed25519")]
    UnsupportedAlgorithm,
    #[error("unsupported registry key-rotation schema version: {0}")]
    UnsupportedSchemaVersion(u32),
    #[error("registry signature key ID must not be empty")]
    EmptyKeyId,
    #[error("registry signature policy contains duplicate key ID: {0}")]
    DuplicateKeyId(String),
    #[error("registry signature key ID is not trusted: {0}")]
    UntrustedKeyId(String),
    #[error("registry signature key ID cannot be revoked because it is not trusted: {0}")]
    UnknownKeyId(String),
    #[error("registry signature rotation would leave no trusted keys")]
    EmptyTrustStore,
    #[error("registry signature public key does not match trusted key ID: {key_id}")]
    PublicKeyMismatch { key_id: String },
    #[error("{field} must contain exactly {expected} hexadecimal characters")]
    InvalidHexLength {
        field: &'static str,
        expected: usize,
    },
    #[error("{field} contains non-hexadecimal data")]
    InvalidHex { field: &'static str },
    #[error("invalid Ed25519 public key: {0}")]
    InvalidPublicKey(String),
    #[error("registry manifest hash failed: {0}")]
    ManifestHash(String),
    #[error("{field} exceeds maximum serialized size of {max} bytes")]
    PayloadTooLarge { field: &'static str, max: usize },
    #[error("registry manifest signature verification failed")]
    VerificationFailed,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryPreviewRequest {
    schema_version: u32,
    modules: Vec<RegistryPreviewModule>,
    #[serde(default)]
    manifest_hash: Option<String>,
    #[serde(default)]
    signature: Option<RegistryManifestSignature>,
}

pub const REGISTRY_PREVIEW_SCHEMA_VERSION: u32 = 1;
/// Version of the JSON envelope returned by canonical-IR preview APIs.
pub const CANONICAL_IR_PREVIEW_SCHEMA_VERSION: u32 = 1;
const MAX_REGISTRY_PREVIEW_BYTES: usize = 8 * 1024 * 1024;
const MAX_REGISTRY_PREVIEW_MODULES: usize = 4096;
const MAX_REGISTRY_TRUST_POLICY_BYTES: usize = 64 * 1024;
const MAX_CANONICAL_IR_PREVIEW_BYTES: usize = 16 * 1024 * 1024;

impl CanonicalIrPreview {
    fn from_frontend(frontend: FrontendProduct) -> Result<Self, CompileError> {
        let canonical_ir_hash = frontend.ir.structural_hash().map_err(|error| {
            CompileError::Internal(format!("failed to hash canonical IR: {error}"))
        })?;
        let sensitivity_manifest =
            serde_json::to_value(&frontend.ir.sensitivity).map_err(|error| {
                CompileError::Internal(format!("failed to serialize sensitivity manifest: {error}"))
            })?;
        let provenance_manifest =
            preview_provenance_manifest_value(&frontend.ast, frontend.ir.as_ref())?;
        Ok(Self {
            canonical_ir: frontend.ir,
            canonical_ir_hash,
            diagnostics: frontend.diagnostics,
            sensitivity_manifest,
            provenance_manifest,
        })
    }

    /// Materialize the preview as its stable, ownership-neutral wire envelope.
    pub fn to_envelope(&self) -> CanonicalIrPreviewEnvelope {
        let diagnostics = self
            .diagnostics
            .iter()
            .map(|diagnostic| CanonicalIrPreviewDiagnostic {
                level: match diagnostic.level {
                    DiagLevel::Error => CanonicalIrPreviewDiagnosticLevel::Error,
                    DiagLevel::Warning => CanonicalIrPreviewDiagnosticLevel::Warning,
                    DiagLevel::Info => CanonicalIrPreviewDiagnosticLevel::Info,
                },
                message: diagnostic.message.clone(),
                span: diagnostic.span,
                hint: diagnostic.hint.clone(),
            })
            .collect();
        CanonicalIrPreviewEnvelope {
            preview_schema: CANONICAL_IR_PREVIEW_SCHEMA_VERSION,
            compiler: "meshlang-compiler".to_owned(),
            compiler_version: env!("CARGO_PKG_VERSION").to_owned(),
            canonical_ir_hash: self.canonical_ir_hash.to_string(),
            canonical_ir: VersionedCanonicalIr::current(self.canonical_ir.as_ref().clone()),
            diagnostics,
            sensitivity_manifest: Some(self.sensitivity_manifest.clone()),
            provenance_manifest: Some(self.provenance_manifest.clone()),
            registry_manifest_hash: None,
            registry_signature_key_id: None,
        }
    }

    /// Serialize the preview as a stable, ownership-neutral JSON envelope.
    pub fn to_json(&self) -> Result<String, CompileError> {
        self.to_json_for_envelope(self.to_envelope())
    }

    fn to_json_for_envelope(
        &self,
        envelope: CanonicalIrPreviewEnvelope,
    ) -> Result<String, CompileError> {
        serde_json::to_string_pretty(&envelope).map_err(|error| {
            CompileError::Internal(format!("failed to serialize canonical IR preview: {error}"))
        })
    }

    fn to_json_with_registry_manifest(
        &self,
        manifest_hash: StructuralHash,
        signature_key_id: Option<&str>,
    ) -> Result<String, CompileError> {
        let mut envelope = self.to_envelope();
        envelope.registry_manifest_hash = Some(manifest_hash.to_string());
        envelope.registry_signature_key_id = signature_key_id.map(str::to_owned);
        self.to_json_for_envelope(envelope)
    }
}

impl CanonicalIrPreviewEnvelope {
    /// Parse and validate a preview artifact received from a host or browser.
    ///
    /// Validation checks both the wire schema and the structural identity of
    /// the embedded IR. The latter is an integrity check for transport and
    /// cache identity; [`StructuralHash`] is deliberately not a cryptographic
    /// signature.
    pub fn from_json(json: &str) -> Result<Self, CompileError> {
        if json.len() > MAX_CANONICAL_IR_PREVIEW_BYTES {
            return Err(CompileError::Unsupported(format!(
                "canonical IR preview payload exceeds {MAX_CANONICAL_IR_PREVIEW_BYTES} bytes"
            )));
        }
        let envelope: Self = serde_json::from_str(json).map_err(|error| {
            CompileError::Parse(format!("invalid canonical IR preview: {error}"))
        })?;
        if envelope.preview_schema != CANONICAL_IR_PREVIEW_SCHEMA_VERSION {
            return Err(CompileError::Unsupported(format!(
                "unsupported canonical IR preview schema {}; expected {}",
                envelope.preview_schema, CANONICAL_IR_PREVIEW_SCHEMA_VERSION
            )));
        }
        if envelope.compiler != "meshlang-compiler" {
            return Err(CompileError::Unsupported(format!(
                "unsupported canonical IR preview compiler `{}`",
                envelope.compiler
            )));
        }
        if let Some(manifest_hash) = &envelope.registry_manifest_hash {
            if manifest_hash.len() != 64
                || !manifest_hash
                    .chars()
                    .all(|character| character.is_ascii_hexdigit())
            {
                return Err(CompileError::Unsupported(
                    "canonical IR preview registry manifest hash is not a 256-bit hexadecimal value"
                        .into(),
                ));
            }
        }
        if envelope.canonical_ir.document_version != CURRENT_IR_DOCUMENT_VERSION {
            return Err(CompileError::Unsupported(format!(
                "unsupported canonical IR document version {}",
                envelope.canonical_ir.document_version.0
            )));
        }
        envelope.canonical_ir.ir.validate().map_err(|errors| {
            CompileError::Unsupported(format!("invalid canonical IR: {errors:?}"))
        })?;
        let expected_hash = envelope
            .canonical_ir
            .ir
            .structural_hash()
            .map_err(|error| {
                CompileError::Internal(format!("failed to hash canonical IR: {error}"))
            })?
            .to_string();
        if envelope.canonical_ir_hash != expected_hash {
            return Err(CompileError::Unsupported(
                "canonical IR preview hash does not match embedded IR".into(),
            ));
        }
        Ok(envelope)
    }
}

/// Compiler state intended to be reused across editor updates and design
/// candidate evaluations.
///
/// Imported modules are invalidated by their source-content hash. Flattened
/// ASTs reuse canonical IR by structural hash, so unchanged candidates avoid
/// lowering while changed candidates cannot observe stale IR.
pub struct IncrementalCompiler {
    options: CompileOptions,
    imports: ImportResolver,
    canonical_ir: CanonicalIrCache,
    /// Backend products are keyed by the complete flattened AST hash rather
    /// than only the canonical IR hash. This keeps source-level constructs
    /// that are still interpreted by a backend (for example comments or
    /// control-flow spelling) from being accidentally reused across revisions.
    backend_artifacts: HashMap<(StructuralHash, BackendKind), CompileResult>,
    backend_artifact_hits: u64,
    backend_artifact_misses: u64,
    artifact_sections: HashMap<StructuralHash, CachedArtifactSections>,
    artifact_section_hits: u64,
    artifact_section_misses: u64,
}

impl IncrementalCompiler {
    /// Construct a compiler rooted at `base_dir` with the default policy.
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Self {
            options: CompileOptions::default(),
            imports: ImportResolver::with_defaults(base_dir.as_ref()),
            canonical_ir: CanonicalIrCache::default(),
            backend_artifacts: HashMap::new(),
            backend_artifact_hits: 0,
            backend_artifact_misses: 0,
            artifact_sections: HashMap::new(),
            artifact_section_hits: 0,
            artifact_section_misses: 0,
        }
    }

    /// Construct a compiler with explicit import and validation policy.
    pub fn with_options(
        base_dir: impl AsRef<Path>,
        options: CompileOptions,
    ) -> Result<Self, CompileError> {
        let mut imports = ImportResolver::with_defaults(base_dir.as_ref());
        for path in &options.import_search_paths {
            imports.add_search_path(path);
        }
        if let Some(path) = &options.stdlib_path {
            imports.set_stdlib_path(path);
        }
        if let Some(root) = &options.import_sandbox_root {
            imports.set_sandbox_root(root)?;
        }
        imports.set_allow_absolute_paths(options.allow_absolute_imports);
        Ok(Self {
            options,
            imports,
            canonical_ir: CanonicalIrCache::default(),
            backend_artifacts: HashMap::new(),
            backend_artifact_hits: 0,
            backend_artifact_misses: 0,
            artifact_sections: HashMap::new(),
            artifact_section_hits: 0,
            artifact_section_misses: 0,
        })
    }

    /// Compile one source revision to GEO while reusing module and IR caches.
    pub fn compile_to_geo(
        &mut self,
        source: &str,
        filename: &str,
    ) -> Result<CompileResult, CompileError> {
        let frontend = self.prepare(source, filename)?;
        let key = (Self::backend_source_hash(&frontend.ast)?, BackendKind::Geo);
        if let Some(cached) = self.backend_artifacts.get(&key) {
            self.backend_artifact_hits += 1;
            return Ok(cached.clone());
        }
        self.backend_artifact_misses += 1;
        let mut codegen = GeoCodegen::new();
        codegen.apply_canonical_plan(&frontend.ir)?;
        super::canonical_ast::validate_backend_capabilities_with_hessian_provider(
            &frontend.ir,
            self.options.hessian_metric_provider.is_some(),
        )?;
        let mut backend_ast =
            super::canonical_ast::materialize_geometry(&frontend.ast, &frontend.ir)?;
        super::canonical_ast::materialize_hessian_metrics(
            &mut backend_ast,
            self.options.hessian_metric_provider.as_deref(),
        )?;
        codegen.emit_source_file(&backend_ast)?;
        codegen.emit_canonical_controls(&frontend.ir)?;
        codegen.validate_canonical_output(&frontend.ir)?;
        codegen
            .tag_allocator()
            .validate_reverse_mappings()
            .map_err(CompileError::Internal)?;
        let tag_map_json = Some(
            serde_json::to_string_pretty(
                &codegen
                    .tag_allocator()
                    .export_map_with_provenance(&frontend.ir),
            )
            .map_err(|error| {
                CompileError::Internal(format!("failed to serialize tag map: {error}"))
            })?,
        );
        let sections = self.artifact_sections_for(&frontend.ir)?;
        let result = build_result_with_cached_sections(
            codegen.output(),
            frontend.diagnostics,
            tag_map_json,
            &frontend.ast,
            &frontend.ir,
            "geo",
            sections,
        )?;
        self.backend_artifacts.insert(key, result.clone());
        Ok(result)
    }

    /// Compile one source revision to Python while reusing module and IR caches.
    pub fn compile_to_python(
        &mut self,
        source: &str,
        filename: &str,
    ) -> Result<CompileResult, CompileError> {
        let frontend = self.prepare(source, filename)?;
        let key = (
            Self::backend_source_hash(&frontend.ast)?,
            BackendKind::Python,
        );
        if let Some(cached) = self.backend_artifacts.get(&key) {
            self.backend_artifact_hits += 1;
            return Ok(cached.clone());
        }
        self.backend_artifact_misses += 1;
        let mut codegen = PythonCodegen::new();
        codegen.apply_canonical_plan(&frontend.ir)?;
        super::canonical_ast::validate_backend_capabilities_with_hessian_provider(
            &frontend.ir,
            self.options.hessian_metric_provider.is_some(),
        )?;
        let mut backend_ast =
            super::canonical_ast::materialize_geometry(&frontend.ast, &frontend.ir)?;
        super::canonical_ast::materialize_hessian_metrics(
            &mut backend_ast,
            self.options.hessian_metric_provider.as_deref(),
        )?;
        codegen.emit_source_file(&backend_ast)?;
        codegen.emit_canonical_controls(&frontend.ir)?;
        codegen.validate_canonical_output(&frontend.ir)?;
        let sections = self.artifact_sections_for(&frontend.ir)?;
        let result = build_result_with_cached_sections(
            codegen.output(),
            frontend.diagnostics,
            None,
            &frontend.ast,
            &frontend.ir,
            "python",
            sections,
        )?;
        self.backend_artifacts.insert(key, result.clone());
        Ok(result)
    }

    /// Run the shared frontend and return canonical IR without backend codegen.
    pub fn preview_canonical_ir(
        &mut self,
        source: &str,
        filename: &str,
    ) -> Result<CanonicalIrPreview, CompileError> {
        CanonicalIrPreview::from_frontend(self.prepare(source, filename)?)
    }

    /// Current cache sizes and cumulative hit/miss counters.
    pub fn stats(&self) -> IncrementalCompilerStats {
        let module_stats = self.imports.cache_stats();
        IncrementalCompilerStats {
            parsed_module_entries: self.imports.cached_module_count(),
            parsed_module_hits: module_stats.hits,
            parsed_module_misses: module_stats.misses,
            canonical_ir_entries: self.canonical_ir.len(),
            canonical_ir_hits: self.canonical_ir.hits(),
            canonical_ir_misses: self.canonical_ir.misses(),
            backend_artifact_entries: self.backend_artifacts.len(),
            backend_artifact_hits: self.backend_artifact_hits,
            backend_artifact_misses: self.backend_artifact_misses,
            artifact_section_entries: self.artifact_sections.len(),
            artifact_section_hits: self.artifact_section_hits,
            artifact_section_misses: self.artifact_section_misses,
        }
    }

    /// Register a validated package index for all subsequent incremental
    /// imports. The resolver performs atomic key preflight and preserves its
    /// exact-version/source-hash cache contract.
    pub fn register_package_index(
        &mut self,
        index: &crate::registry::RegistryPackageIndex,
    ) -> Result<(), CompileError> {
        self.imports.register_package_index(index)
    }

    /// Register a package index with an explicit content-pinned migration
    /// policy. The default facade remains exact-version-only.
    pub fn register_package_index_with_migration_policy(
        &mut self,
        index: &crate::registry::RegistryPackageIndex,
        migration_policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<(), CompileError> {
        self.imports
            .register_package_index_with_migration_policy(index, migration_policy)
    }

    /// Verify and register a package index through the host-owned trust
    /// policy, returning the canonical authenticated index hash.
    pub fn register_trusted_package_index(
        &mut self,
        envelope: &crate::registry::RegistryPackageIndexEnvelope,
        policy: &crate::incremental::RegistrySignaturePolicy,
    ) -> Result<StructuralHash, CompileError> {
        self.imports
            .register_trusted_package_index(envelope, policy)
    }

    /// Verify and register a signed package index with an explicit migration
    /// policy before any resolver mutation.
    pub fn register_trusted_package_index_with_migration_policy(
        &mut self,
        envelope: &crate::registry::RegistryPackageIndexEnvelope,
        policy: &RegistrySignaturePolicy,
        migration_policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<StructuralHash, CompileError> {
        self.imports
            .register_trusted_package_index_with_migration_policy(
                envelope,
                policy,
                migration_policy,
            )
    }

    /// Fetch and trust one exact package version through a host-owned
    /// transport, then register its canonical module set atomically.
    pub fn register_trusted_package_from_transport<T>(
        &mut self,
        package: &str,
        version: &str,
        transport: &mut T,
        policy: &RegistrySignaturePolicy,
    ) -> Result<StructuralHash, CompileError>
    where
        T: crate::registry::RegistryPackageTransport,
    {
        self.register_trusted_package_from_transport_with_migration_policy(
            package,
            version,
            transport,
            policy,
            &crate::registry::RegistryPackageMigrationPolicy::Reject,
        )
    }

    /// Fetch and trust one exact package with an explicit migration policy.
    pub fn register_trusted_package_from_transport_with_migration_policy<T>(
        &mut self,
        package: &str,
        version: &str,
        transport: &mut T,
        policy: &RegistrySignaturePolicy,
        migration_policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<StructuralHash, CompileError>
    where
        T: crate::registry::RegistryPackageTransport,
    {
        self.imports
            .register_trusted_package_from_transport_with_migration_policy(
                package,
                version,
                transport,
                policy,
                migration_policy,
            )
    }

    /// Discover a package version through a host-owned catalog, fetch the
    /// selected exact version through trusted transport, and register it
    /// atomically for subsequent incremental imports.
    pub fn register_trusted_package_from_discovery<D, T>(
        &mut self,
        package: &str,
        version_policy: &crate::registry::RegistryPackageVersionPolicy,
        discovery: &mut D,
        transport: &mut T,
        trust: &RegistrySignaturePolicy,
    ) -> Result<crate::registry::RegistryPackageResolution, CompileError>
    where
        D: crate::registry::RegistryPackageDiscovery,
        T: crate::registry::RegistryPackageTransport,
    {
        self.register_trusted_package_from_discovery_with_migration_policy(
            package,
            version_policy,
            discovery,
            transport,
            trust,
            &crate::registry::RegistryPackageMigrationPolicy::Reject,
        )
    }

    /// Discover and register one package with an explicit migration policy.
    pub fn register_trusted_package_from_discovery_with_migration_policy<D, T>(
        &mut self,
        package: &str,
        version_policy: &crate::registry::RegistryPackageVersionPolicy,
        discovery: &mut D,
        transport: &mut T,
        trust: &RegistrySignaturePolicy,
        migration_policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<crate::registry::RegistryPackageResolution, CompileError>
    where
        D: crate::registry::RegistryPackageDiscovery,
        T: crate::registry::RegistryPackageTransport,
    {
        self.imports
            .register_trusted_package_from_discovery_with_migration_policy(
                package,
                version_policy,
                discovery,
                transport,
                trust,
                migration_policy,
            )
    }

    /// Invalidate one imported module. Source hashing still protects callers
    /// that do not issue explicit invalidations.
    pub fn invalidate_module(&mut self, path: impl AsRef<Path>) -> bool {
        self.imports.invalidate(path)
    }

    /// Release cached modules and IR variants while retaining stable file IDs.
    pub fn clear_caches(&mut self) {
        self.imports.clear_cache();
        self.canonical_ir.clear();
        self.backend_artifacts.clear();
        self.artifact_sections.clear();
    }

    /// Reset cumulative cache counters without discarding entries.
    pub fn reset_stats(&mut self) {
        self.imports.reset_cache_stats();
        self.canonical_ir.reset_stats();
        self.backend_artifact_hits = 0;
        self.backend_artifact_misses = 0;
        self.artifact_section_hits = 0;
        self.artifact_section_misses = 0;
    }

    pub fn options(&self) -> &CompileOptions {
        &self.options
    }

    fn prepare(&mut self, source: &str, filename: &str) -> Result<FrontendProduct, CompileError> {
        let ast = meshlang_parser::parse(source, filename)
            .map_err(|error| CompileError::Parse(error.message))?;
        let ast = flatten_imports_with_resolver(ast, filename, &mut self.imports)?;
        analyze_flattened_ast(ast, &self.options, &mut self.canonical_ir)
    }

    fn backend_source_hash(file: &SourceFile) -> Result<StructuralHash, CompileError> {
        file.structural_hash().map_err(|error| {
            CompileError::Internal(format!(
                "failed to hash flattened AST for backend cache: {error}"
            ))
        })
    }

    fn artifact_sections_for(
        &mut self,
        ir: &CanonicalGeometryIr,
    ) -> Result<CachedArtifactSections, CompileError> {
        let key = ir.structural_hash().map_err(|error| {
            CompileError::Internal(format!("failed to hash canonical IR: {error}"))
        })?;
        if let Some(cached) = self.artifact_sections.get(&key) {
            self.artifact_section_hits += 1;
            return Ok(cached.clone());
        }
        self.artifact_section_misses += 1;
        let sections = build_artifact_sections(ir)?;
        self.artifact_sections.insert(key, sections.clone());
        Ok(sections)
    }
}

/// Single-source canonical-IR preview. MeshLang imports are deliberately
/// rejected because this API is also the deterministic browser/WASM surface;
/// native tools with a module graph should use [`IncrementalCompiler`].
pub fn preview_canonical_ir(
    source: &str,
    filename: &str,
) -> Result<CanonicalIrPreview, CompileError> {
    preview_canonical_ir_with_options(source, filename, &CompileOptions::default())
}

pub fn preview_canonical_ir_with_options(
    source: &str,
    filename: &str,
    options: &CompileOptions,
) -> Result<CanonicalIrPreview, CompileError> {
    let ast = meshlang_parser::parse(source, filename)
        .map_err(|error| CompileError::Parse(error.message))?;
    if ast.declarations.iter().any(|declaration| {
        matches!(declaration, Declaration::Import(import) if crate::imports::is_meshlang_import(&import.path))
    }) {
        return Err(CompileError::Unsupported(
            "single-source canonical IR preview does not resolve MeshLang imports; use IncrementalCompiler"
                .into(),
        ));
    }
    let mut cache = CanonicalIrCache::default();
    CanonicalIrPreview::from_frontend(analyze_flattened_ast(ast, options, &mut cache)?)
}

pub fn preview_canonical_ir_json(source: &str, filename: &str) -> Result<String, CompileError> {
    preview_canonical_ir(source, filename)?.to_json()
}

/// Preview a source file with an explicit, filesystem-independent registry.
///
/// This is the native implementation behind the WASM/editor API. The
/// versioned JSON request keeps registry transport deterministic and bounds
/// untrusted payloads before parsing or lowering. Filesystem imports are
/// rejected even when a same-named host file happens to exist.
pub fn preview_canonical_ir_with_registry_json(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, CompileError> {
    preview_canonical_ir_with_registry_json_internal(source, filename, registry_json, false, None)
}

/// Preview a source file against a signed, filesystem-independent registry.
///
/// This entry point fails closed when the request has no Ed25519 signature.
/// The unsigned API remains available for local cache-integrity use, while
/// browser/package consumers can opt into authenticity verification explicitly.
pub fn preview_canonical_ir_with_signed_registry_json(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, CompileError> {
    preview_canonical_ir_with_registry_json_internal(source, filename, registry_json, true, None)
}

/// Preview a source file against a signed registry and a host trust policy.
pub fn preview_canonical_ir_with_trusted_registry_json(
    source: &str,
    filename: &str,
    registry_json: &str,
    trust_policy_json: &str,
) -> Result<String, CompileError> {
    if trust_policy_json.len() > MAX_REGISTRY_TRUST_POLICY_BYTES {
        return Err(CompileError::Unsupported(format!(
            "registry trust policy exceeds {MAX_REGISTRY_TRUST_POLICY_BYTES} bytes"
        )));
    }
    let policy: RegistrySignaturePolicy =
        serde_json::from_str(trust_policy_json).map_err(|error| {
            CompileError::Parse(format!("invalid registry trust policy JSON: {error}"))
        })?;
    policy.validate().map_err(|error| {
        CompileError::Unsupported(format!("registry trust policy rejected: {error}"))
    })?;
    preview_canonical_ir_with_registry_json_internal(
        source,
        filename,
        registry_json,
        true,
        Some(&policy),
    )
}

/// Preview a source file against an authenticated package-index envelope.
///
/// Hosts fetch the envelope (for example through HTTP or an embedded asset)
/// and pass the bytes across this filesystem-free boundary. The package index
/// is authenticated against the caller's pinned trust policy, then registered
/// atomically in registry-only mode before imports are flattened. This keeps
/// package transport host-owned while making the signed package-index contract
/// directly consumable by browser/editor clients.
pub fn preview_canonical_ir_with_trusted_package_json(
    source: &str,
    filename: &str,
    package_json: &str,
    trust_policy_json: &str,
) -> Result<String, CompileError> {
    if package_json.len() > MAX_REGISTRY_PREVIEW_BYTES {
        return Err(CompileError::Unsupported(format!(
            "signed package index exceeds {MAX_REGISTRY_PREVIEW_BYTES} bytes"
        )));
    }
    if trust_policy_json.len() > MAX_REGISTRY_TRUST_POLICY_BYTES {
        return Err(CompileError::Unsupported(format!(
            "registry trust policy exceeds {MAX_REGISTRY_TRUST_POLICY_BYTES} bytes"
        )));
    }
    let policy: RegistrySignaturePolicy =
        serde_json::from_str(trust_policy_json).map_err(|error| {
            CompileError::Parse(format!("invalid registry trust policy JSON: {error}"))
        })?;
    policy.validate().map_err(|error| {
        CompileError::Unsupported(format!("registry trust policy rejected: {error}"))
    })?;
    let envelope = crate::registry::RegistryPackageIndexEnvelope::from_json(package_json)?;
    let index_hash = envelope.verify_with_policy(&policy).map_err(|error| {
        CompileError::Unsupported(format!("package-index trust policy rejected: {error}"))
    })?;
    let signature_key_id = envelope.signature.key_id.clone();

    let ast = meshlang_parser::parse(source, filename)
        .map_err(|error| CompileError::Parse(error.message))?;
    let mut resolver = ImportResolver::new();
    resolver.set_registry_only(true);
    resolver.register_trusted_package_index(&envelope, &policy)?;
    let ast = flatten_imports_with_resolver(ast, filename, &mut resolver)?;
    let mut ir_cache = CanonicalIrCache::default();
    let frontend = analyze_flattened_ast(ast, &CompileOptions::default(), &mut ir_cache)?;
    CanonicalIrPreview::from_frontend(frontend)?
        .to_json_with_registry_manifest(index_hash, Some(&signature_key_id))
}

fn preview_canonical_ir_with_registry_json_internal(
    source: &str,
    filename: &str,
    registry_json: &str,
    require_signature: bool,
    policy: Option<&RegistrySignaturePolicy>,
) -> Result<String, CompileError> {
    if registry_json.len() > MAX_REGISTRY_PREVIEW_BYTES {
        return Err(CompileError::Unsupported(format!(
            "registry preview payload exceeds {MAX_REGISTRY_PREVIEW_BYTES} bytes"
        )));
    }
    let request: RegistryPreviewRequest = serde_json::from_str(registry_json)
        .map_err(|error| CompileError::Parse(format!("invalid registry preview JSON: {error}")))?;
    if request.schema_version != REGISTRY_PREVIEW_SCHEMA_VERSION {
        return Err(CompileError::Unsupported(format!(
            "unsupported registry preview schema version {}",
            request.schema_version
        )));
    }
    if request.modules.len() > MAX_REGISTRY_PREVIEW_MODULES {
        return Err(CompileError::Unsupported(format!(
            "registry preview contains {} modules, maximum is {MAX_REGISTRY_PREVIEW_MODULES}",
            request.modules.len()
        )));
    }
    let manifest_hash = registry_preview_manifest_hash(&request.modules)?;
    if let Some(expected) = request.manifest_hash.as_deref() {
        if expected != manifest_hash.to_string() {
            return Err(CompileError::Unsupported(
                "registry preview manifest hash does not match supplied modules".into(),
            ));
        }
    }

    let signature_key_id = if let Some(signature) = request.signature.as_ref() {
        let verify = policy.map_or_else(
            || verify_registry_manifest_signature(&request.modules, signature),
            |policy| policy.verify(&request.modules, signature),
        );
        verify.map_err(|error| {
            CompileError::Unsupported(format!("registry manifest signature rejected: {error}"))
        })?;
        Some(signature.key_id.as_str())
    } else if require_signature {
        return Err(CompileError::Unsupported(
            "signed registry preview requires a manifest signature".into(),
        ));
    } else {
        None
    };
    let mut resolver = ImportResolver::new();
    resolver.set_registry_only(true);
    for module in request.modules {
        resolver.register_module(module.path, module.source)?;
    }
    let ast = meshlang_parser::parse(source, filename)
        .map_err(|error| CompileError::Parse(error.message))?;
    let ast = crate::flatten_imports_with_resolver(ast, filename, &mut resolver)?;
    let mut cache = CanonicalIrCache::default();
    let preview = CanonicalIrPreview::from_frontend(analyze_flattened_ast(
        ast,
        &CompileOptions::default(),
        &mut cache,
    )?)?;
    preview.to_json_with_registry_manifest(manifest_hash, signature_key_id)
}

/// Compute an order-independent structural identity for a source registry.
///
/// The hash is suitable for cache keys and transport integrity checks. It is
/// deliberately not a cryptographic signature and must not be used as one.
pub fn registry_preview_manifest_hash(
    modules: &[RegistryPreviewModule],
) -> Result<StructuralHash, CompileError> {
    let mut entries = modules
        .iter()
        .map(|module| (&module.path, &module.source))
        .collect::<Vec<_>>();
    entries.sort();
    StructuralHash::of(&entries).map_err(|error| {
        CompileError::Internal(format!("failed to hash registry manifest: {error}"))
    })
}

/// Build the exact domain-separated bytes covered by a registry signature.
///
/// The payload includes the schema version and key ID in addition to the
/// order-independent manifest hash, preventing replay across registry
/// protocols or under another key identity.
pub fn registry_manifest_signing_payload(
    modules: &[RegistryPreviewModule],
    key_id: &str,
) -> Result<Vec<u8>, CompileError> {
    if key_id.trim().is_empty() {
        return Err(CompileError::Unsupported(
            "registry signature key ID must not be empty".into(),
        ));
    }
    let manifest_hash = registry_preview_manifest_hash(modules)?;
    Ok(format!(
        "sonium-meshlang-registry-manifest-v1\n{}\n{}\n{}",
        REGISTRY_PREVIEW_SCHEMA_VERSION, key_id, manifest_hash
    )
    .into_bytes())
}

/// Verify a detached Ed25519 signature for a canonical source registry.
pub fn verify_registry_manifest_signature(
    modules: &[RegistryPreviewModule],
    signature: &RegistryManifestSignature,
) -> Result<StructuralHash, RegistrySignatureError> {
    if signature.algorithm != "ed25519" {
        return Err(RegistrySignatureError::UnsupportedAlgorithm);
    }
    if signature.key_id.trim().is_empty() {
        return Err(RegistrySignatureError::EmptyKeyId);
    }
    let public_key = decode_hex::<32>(&signature.public_key, "public_key")?;
    let signature_bytes = decode_hex::<64>(&signature.signature, "signature")?;
    let manifest_hash = registry_preview_manifest_hash(modules)
        .map_err(|error| RegistrySignatureError::ManifestHash(error.to_string()))?;
    let payload = registry_manifest_signing_payload(modules, &signature.key_id)
        .map_err(|error| RegistrySignatureError::ManifestHash(error.to_string()))?;
    let verification_key = VerificationKey::try_from(VerificationKeyBytes::from(public_key))
        .map_err(|error| RegistrySignatureError::InvalidPublicKey(format!("{error:?}")))?;
    let detached = Signature::from_bytes(&signature_bytes);
    verification_key
        .verify(&detached, &payload)
        .map_err(|_| RegistrySignatureError::VerificationFailed)?;
    Ok(manifest_hash)
}

fn decode_hex<const N: usize>(
    value: &str,
    field: &'static str,
) -> Result<[u8; N], RegistrySignatureError> {
    if value.len() != N * 2 {
        return Err(RegistrySignatureError::InvalidHexLength {
            field,
            expected: N * 2,
        });
    }
    let bytes = value.as_bytes();
    let mut decoded = [0_u8; N];
    for (index, chunk) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(RegistrySignatureError::InvalidHex { field })?;
        let low = hex_nibble(chunk[1]).ok_or(RegistrySignatureError::InvalidHex { field })?;
        decoded[index] = (high << 4) | low;
    }
    Ok(decoded)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Browser/editor entry point. JSON avoids JavaScript ownership ambiguity and
/// matches the native preview envelope exactly.
#[cfg(all(target_arch = "wasm32", feature = "wasm-exports"))]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIr)]
pub fn preview_canonical_ir_wasm(
    source: &str,
    filename: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_canonical_ir_json(source, filename)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Browser/editor entry point with an explicit source registry.
#[cfg(all(target_arch = "wasm32", feature = "wasm-exports"))]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIrWithRegistry)]
pub fn preview_canonical_ir_with_registry_wasm(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_canonical_ir_with_registry_json(source, filename, registry_json)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Browser/editor entry point requiring an authenticated source registry.
#[cfg(all(target_arch = "wasm32", feature = "wasm-exports"))]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIrWithSignedRegistry)]
pub fn preview_canonical_ir_with_signed_registry_wasm(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_canonical_ir_with_signed_registry_json(source, filename, registry_json)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Browser/editor entry point requiring an authenticated registry and pinned
/// trust anchors supplied as a versioned JSON policy.
#[cfg(all(target_arch = "wasm32", feature = "wasm-exports"))]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIrWithTrustedRegistry)]
pub fn preview_canonical_ir_with_trusted_registry_wasm(
    source: &str,
    filename: &str,
    registry_json: &str,
    trust_policy_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_canonical_ir_with_trusted_registry_json(
        source,
        filename,
        registry_json,
        trust_policy_json,
    )
    .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
