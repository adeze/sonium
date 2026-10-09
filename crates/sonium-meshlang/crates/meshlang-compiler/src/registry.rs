//! Deterministic local package-index policy for MeshLang registries.
//!
//! This module deliberately stops at an authenticated, host-independent index
//! boundary. It does not fetch packages or resolve network URLs. Callers can
//! verify an index, select an exact package version, and then hand the module
//! sources to the crate import resolver (or sign the canonical module list
//! with the existing registry-signature API).

use ed25519_zebra::{Signature, VerificationKey, VerificationKeyBytes};
use meshlang_ast::StructuralHash;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::incremental::RegistryKeyRotationEnvelope;
use crate::CompileError;

/// Version of the package-index JSON envelope.
pub const REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION: u32 = 1;
/// Version of the explicit package-migration plan JSON envelope.
pub const REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION: u32 = 1;
const MAX_PACKAGE_INDEX_BYTES: usize = 8 * 1024 * 1024;
const MAX_PACKAGES: usize = 1024;
const MAX_MODULES_PER_PACKAGE: usize = 4096;
const MAX_DEPENDENCIES_PER_PACKAGE: usize = 1024;
const MAX_MIGRATION_MODULES: usize = 4096;

/// A source-backed module published by one exact package version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageModule {
    /// Canonical slash-separated path, rooted at the package name.
    pub path: String,
    pub source: String,
}

/// An exact package dependency. Version ranges are intentionally not accepted
/// so two hosts cannot select different modules from the same index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageDependency {
    pub name: String,
    pub version: String,
    /// Optional exact content pin for the dependency manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_hash: Option<String>,
}

/// One exact package version and its complete source/dependency contents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageManifest {
    pub name: String,
    pub version: String,
    pub modules: Vec<RegistryPackageModule>,
    #[serde(default)]
    pub dependencies: Vec<RegistryPackageDependency>,
    /// Structural hash of the canonical package fields above.
    pub manifest_hash: String,
}

/// Versioned, self-contained package index suitable for local/WASM transport.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageIndex {
    pub schema_version: u32,
    pub packages: Vec<RegistryPackageManifest>,
}

/// Detached Ed25519 signature over one canonical package index.
///
/// The package-index payload has its own domain separator and includes the
/// schema version, key ID, and canonical index hash. It is therefore not
/// interchangeable with a source-registry module signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageIndexSignature {
    pub algorithm: String,
    pub key_id: String,
    /// Lowercase hexadecimal Ed25519 public key (32 bytes).
    pub public_key: String,
    /// Lowercase hexadecimal Ed25519 signature (64 bytes).
    pub signature: String,
}

/// Authenticated package-index JSON envelope for transport to hosts/WASM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageIndexEnvelope {
    pub index: RegistryPackageIndex,
    pub signature: RegistryPackageIndexSignature,
}

/// Host-owned transport failure while fetching an exact package-index
/// envelope. The compiler deliberately does not perform network I/O; hosts,
/// package managers, or WASM adapters implement this trait and decide how a
/// package is retrieved and authenticated in transit.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryPackageTransportError {
    #[error("invalid package request: {0}")]
    InvalidRequest(String),
    #[error("package transport unavailable: {0}")]
    Unavailable(String),
    #[error("invalid package response: {0}")]
    InvalidResponse(String),
}

/// Host-owned failure while discovering available exact package versions.
///
/// Discovery is deliberately separate from authenticated package transport:
/// the returned version list is only a selection hint. The selected package
/// still has to cross [`RegistryPackageTransport`] and pass the pinned
/// signature policy before any module is registered.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryPackageDiscoveryError {
    /// The discovery request was rejected by the host before I/O.
    #[error("invalid package discovery request: {0}")]
    InvalidRequest(String),
    /// The host could not reach its package catalog.
    #[error("package discovery unavailable: {0}")]
    Unavailable(String),
    /// The host returned a malformed or contradictory version list.
    #[error("invalid package discovery response: {0}")]
    InvalidResponse(String),
}

/// Bounded failure while delivering an authenticated trust-store rotation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryKeyTransportError {
    #[error("invalid key-rotation request: {0}")]
    InvalidRequest(String),
    #[error("key-rotation transport unavailable: {0}")]
    Unavailable(String),
    #[error("invalid key-rotation response: {0}")]
    InvalidResponse(String),
    #[error("key-rotation signature rejected: {0}")]
    Signature(String),
}

/// Host-owned transport for one exact authenticated key rotation.
pub trait RegistryKeyTransport {
    fn fetch_key_rotation(
        &mut self,
        rotation_id: &str,
    ) -> Result<RegistryKeyRotationEnvelope, RegistryKeyTransportError>;
}

fn validate_rotation_id(rotation_id: &str) -> Result<(), RegistryKeyTransportError> {
    if rotation_id.is_empty()
        || rotation_id == "."
        || rotation_id == ".."
        || rotation_id.contains('/')
        || rotation_id.contains('\\')
        || !rotation_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character))
    {
        return Err(RegistryKeyTransportError::InvalidRequest(format!(
            "invalid key-rotation selector '{rotation_id}'"
        )));
    }
    Ok(())
}

const MAX_KEY_ROTATION_BYTES: usize = crate::incremental::MAX_REGISTRY_KEY_ROTATION_BYTES;

/// Deterministic filesystem key-rotation store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemRegistryKeyTransport {
    root: PathBuf,
}

impl FilesystemRegistryKeyTransport {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, rotation_id: &str) -> Result<PathBuf, RegistryKeyTransportError> {
        validate_rotation_id(rotation_id)?;
        let root = fs::canonicalize(&self.root).map_err(|error| {
            RegistryKeyTransportError::Unavailable(format!(
                "cannot canonicalize key-rotation root '{}': {error}",
                self.root.display()
            ))
        })?;
        let path = root.join(format!("{rotation_id}.json"));
        if !path.starts_with(&root) {
            return Err(RegistryKeyTransportError::InvalidRequest(
                "key-rotation path escapes configured root".into(),
            ));
        }
        Ok(path)
    }
}

impl RegistryKeyTransport for FilesystemRegistryKeyTransport {
    fn fetch_key_rotation(
        &mut self,
        rotation_id: &str,
    ) -> Result<RegistryKeyRotationEnvelope, RegistryKeyTransportError> {
        let path = self.path(rotation_id)?;
        let metadata = fs::metadata(&path).map_err(|error| {
            RegistryKeyTransportError::Unavailable(format!(
                "cannot inspect key rotation '{}': {error}",
                path.display()
            ))
        })?;
        if metadata.len() > MAX_KEY_ROTATION_BYTES as u64 {
            return Err(RegistryKeyTransportError::InvalidResponse(format!(
                "key rotation '{rotation_id}' exceeds {MAX_KEY_ROTATION_BYTES} bytes"
            )));
        }
        let json = fs::read_to_string(&path).map_err(|error| {
            RegistryKeyTransportError::Unavailable(format!(
                "cannot read key rotation '{}': {error}",
                path.display()
            ))
        })?;
        RegistryKeyRotationEnvelope::from_json(&json)
            .map_err(|error| RegistryKeyTransportError::InvalidResponse(error.to_string()))
    }
}

/// Native HTTP key-rotation transport. WASM callers provide their own
/// `RegistryKeyTransport` implementation, preserving filesystem/network-free
/// compiler APIs on the browser boundary.
#[cfg(not(target_arch = "wasm32"))]
pub struct HttpRegistryKeyTransport {
    base_url: String,
    agent: ureq::Agent,
}

#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Debug for HttpRegistryKeyTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HttpRegistryKeyTransport")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl HttpRegistryKeyTransport {
    pub fn new(base_url: impl Into<String>) -> Result<Self, RegistryKeyTransportError> {
        Self::with_timeout(base_url, std::time::Duration::from_secs(30))
    }

    pub fn with_timeout(
        base_url: impl Into<String>,
        timeout: std::time::Duration,
    ) -> Result<Self, RegistryKeyTransportError> {
        let base_url = base_url.into().trim_end_matches('/').to_owned();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://"))
            || base_url.chars().any(char::is_whitespace)
            || base_url.contains('?')
            || base_url.contains('#')
            || base_url.len() <= "http://".len()
        {
            return Err(RegistryKeyTransportError::InvalidRequest(
                "base URL must be an absolute HTTP(S) URL without query or fragment".into(),
            ));
        }
        if timeout.is_zero() {
            return Err(RegistryKeyTransportError::InvalidRequest(
                "HTTP timeout must be positive".into(),
            ));
        }
        Ok(Self {
            base_url,
            agent: ureq::AgentBuilder::new().timeout(timeout).build(),
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RegistryKeyTransport for HttpRegistryKeyTransport {
    fn fetch_key_rotation(
        &mut self,
        rotation_id: &str,
    ) -> Result<RegistryKeyRotationEnvelope, RegistryKeyTransportError> {
        validate_rotation_id(rotation_id)?;
        let url = format!("{}/{rotation_id}.json", self.base_url);
        let response = self.agent.get(&url).call().map_err(|error| {
            RegistryKeyTransportError::Unavailable(format!(
                "HTTP key rotation '{rotation_id}' request failed: {error}"
            ))
        })?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take((MAX_KEY_ROTATION_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| RegistryKeyTransportError::Unavailable(error.to_string()))?;
        if bytes.len() > MAX_KEY_ROTATION_BYTES {
            return Err(RegistryKeyTransportError::InvalidResponse(format!(
                "HTTP key rotation '{rotation_id}' exceeds {MAX_KEY_ROTATION_BYTES} bytes"
            )));
        }
        let json = String::from_utf8(bytes).map_err(|error| {
            RegistryKeyTransportError::InvalidResponse(format!(
                "HTTP key rotation '{rotation_id}' is not UTF-8: {error}"
            ))
        })?;
        RegistryKeyRotationEnvelope::from_json(&json)
            .map_err(|error| RegistryKeyTransportError::InvalidResponse(error.to_string()))
    }
}

/// Host-owned discovery of versions available for one package name.
///
/// Implementations may use a local cache, an HTTP index, or an embedded
/// catalog. They must not be trusted to provide source contents: the
/// compiler validates the candidates, selects deterministically, then calls
/// [`RegistryPackageTransport`] for the exact selected version.
pub trait RegistryPackageDiscovery {
    /// Return candidate exact numeric dotted versions for `package`.
    fn discover_package_versions(
        &mut self,
        package: &str,
    ) -> Result<Vec<String>, RegistryPackageDiscoveryError>;
}

/// Explicit policy for selecting one version from a discovered catalog.
///
/// No implicit compatibility or migration is performed. `Exact` is the
/// default-safe policy; `HighestAtMost` is an explicit caller opt-in for a
/// deterministic downgrade/upgrade within a numeric dotted bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryPackageVersionPolicy {
    /// Select exactly this numeric dotted version.
    Exact(String),
    /// Select the numerically highest discovered version not above the bound.
    HighestAtMost(String),
}

impl RegistryPackageVersionPolicy {
    /// Validate the selector without performing discovery or transport I/O.
    pub fn validate(&self) -> Result<(), CompileError> {
        match self {
            Self::Exact(version) | Self::HighestAtMost(version) => {
                validate_package_version(version)
            }
        }
    }
}

/// One exact module-path rewrite in an explicitly approved package migration.
///
/// The old path is never treated as an implicit fallback: callers must pass a
/// [`RegistryPackageMigrationPolicy::AllowDeclared`] plan to opt into these
/// rewrites.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageMigrationModule {
    pub from: String,
    pub to: String,
}

/// Content-pinned migration from one exact package version to another.
///
/// A plan is deliberately kept outside [`RegistryPackageIndex`] so existing
/// index JSON remains schema-compatible. The two manifest hashes bind the
/// plan to the authenticated index selected by the host; every module in the
/// source package must be mapped exactly once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageMigrationPlan {
    pub schema_version: u32,
    pub package: String,
    pub from_version: String,
    pub to_version: String,
    pub from_manifest_hash: String,
    pub to_manifest_hash: String,
    pub modules: Vec<RegistryPackageMigrationModule>,
}

/// Explicit policy for applying a package migration plan.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RegistryPackageMigrationPolicy {
    /// Preserve exact import resolution and reject migration aliases.
    #[default]
    Reject,
    /// Apply one validated, content-pinned migration plan.
    AllowDeclared(RegistryPackageMigrationPlan),
}

/// Canonical package modules together with explicitly declared migration
/// aliases.
pub type RegistryPackageModulesWithAliases = (Vec<RegistryPackageModule>, Vec<(String, String)>);

impl RegistryPackageMigrationPlan {
    /// Validate this plan against one canonical package index and return a
    /// deterministic module alias list `(old_path, new_path)`.
    pub fn validate_against(
        &self,
        index: &RegistryPackageIndex,
    ) -> Result<Vec<(String, String)>, CompileError> {
        if self.schema_version != REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION {
            return Err(CompileError::Unsupported(format!(
                "unsupported registry package migration schema {}; expected {}",
                self.schema_version, REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION
            )));
        }
        validate_package_name(&self.package)?;
        validate_package_version(&self.from_version)?;
        validate_package_version(&self.to_version)?;
        if compare_versions(&self.from_version, &self.to_version) != std::cmp::Ordering::Less {
            return Err(CompileError::Unsupported(
                "package migration must move from an older to a newer exact version".into(),
            ));
        }
        validate_hash(&self.from_manifest_hash, "migration from_manifest_hash")?;
        validate_hash(&self.to_manifest_hash, "migration to_manifest_hash")?;
        if self.modules.is_empty() || self.modules.len() > MAX_MIGRATION_MODULES {
            return Err(CompileError::Unsupported(format!(
                "package migration must contain 1..={MAX_MIGRATION_MODULES} module mappings"
            )));
        }

        let canonical = index.clone().validate_and_canonicalize()?;
        let from = canonical
            .resolve(&self.package, &self.from_version)
            .ok_or_else(|| {
                CompileError::Unsupported(format!(
                    "migration source package '{}@{}' is missing from the index",
                    self.package, self.from_version
                ))
            })?;
        let to = canonical
            .resolve(&self.package, &self.to_version)
            .ok_or_else(|| {
                CompileError::Unsupported(format!(
                    "migration target package '{}@{}' is missing from the index",
                    self.package, self.to_version
                ))
            })?;
        if from.manifest_hash != self.from_manifest_hash {
            return Err(CompileError::Unsupported(
                "migration source manifest hash does not match the index".into(),
            ));
        }
        if to.manifest_hash != self.to_manifest_hash {
            return Err(CompileError::Unsupported(
                "migration target manifest hash does not match the index".into(),
            ));
        }

        let source_paths = from
            .modules
            .iter()
            .map(|module| module.path.as_str())
            .collect::<BTreeSet<_>>();
        let target_paths = to
            .modules
            .iter()
            .map(|module| module.path.as_str())
            .collect::<BTreeSet<_>>();
        let mut aliases = self
            .modules
            .iter()
            .map(|mapping| {
                validate_module_path(&self.package, &self.from_version, &mapping.from)?;
                validate_module_path(&self.package, &self.to_version, &mapping.to)?;
                if !source_paths.contains(mapping.from.as_str()) {
                    return Err(CompileError::Unsupported(format!(
                        "migration source module '{}' is missing from the index",
                        mapping.from
                    )));
                }
                if !target_paths.contains(mapping.to.as_str()) {
                    return Err(CompileError::Unsupported(format!(
                        "migration target module '{}' is missing from the index",
                        mapping.to
                    )));
                }
                Ok((mapping.from.clone(), mapping.to.clone()))
            })
            .collect::<Result<Vec<_>, CompileError>>()?;
        aliases.sort();
        if aliases.windows(2).any(|window| window[0].0 == window[1].0) {
            return Err(CompileError::Unsupported(
                "package migration contains duplicate source modules".into(),
            ));
        }
        if aliases.windows(2).any(|window| window[0].1 == window[1].1) {
            return Err(CompileError::Unsupported(
                "package migration contains duplicate target modules".into(),
            ));
        }
        if aliases.len() != source_paths.len()
            || aliases
                .iter()
                .any(|(source, _)| !source_paths.contains(source.as_str()))
        {
            return Err(CompileError::Unsupported(
                "package migration must map every source module exactly once".into(),
            ));
        }
        Ok(aliases)
    }
}

/// Authenticated result of discovery, deterministic selection, and transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryPackageResolution {
    /// Package name requested by the caller.
    pub package: String,
    /// Exact version selected from discovery candidates.
    pub version: String,
    /// Canonical hash of the trusted package index returned by transport.
    pub index_hash: StructuralHash,
}

/// Exact-version package-index transport supplied by a host.
///
/// Implementations may use HTTP, a local package cache, an embedded asset, or
/// another transport. The returned envelope is still checked against the
/// caller's [`RegistrySignaturePolicy`] before any module enters the resolver.
pub trait RegistryPackageTransport {
    fn fetch_package_index(
        &mut self,
        package: &str,
        version: &str,
    ) -> Result<RegistryPackageIndexEnvelope, RegistryPackageTransportError>;
}

/// Filesystem-backed package-index transport and version discovery.
///
/// The store uses the deterministic layout `<root>/<package>/<version>.json`.
/// Package and version selectors are validated before any path is constructed,
/// so a caller cannot escape `root` through traversal or separator characters.
/// Every fetched envelope is parsed and structurally validated before it is
/// returned; signature trust remains the responsibility of the resolver's
/// `RegistrySignaturePolicy`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemPackageRegistry {
    root: PathBuf,
}

impl FilesystemPackageRegistry {
    /// Create a registry rooted at `root` without touching the filesystem.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Return the configured package-store root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn package_directory(&self, package: &str) -> Result<PathBuf, RegistryPackageTransportError> {
        validate_package_name(package)
            .map_err(|error| RegistryPackageTransportError::InvalidRequest(error.to_string()))?;
        Ok(self.root.join(package))
    }

    fn package_index_path(
        &self,
        package: &str,
        version: &str,
    ) -> Result<PathBuf, RegistryPackageTransportError> {
        validate_package_selector(package, version)
            .map_err(|error| RegistryPackageTransportError::InvalidRequest(error.to_string()))?;
        Ok(self.root.join(package).join(format!("{version}.json")))
    }

    fn canonical_path_within_root(
        &self,
        path: &Path,
    ) -> Result<PathBuf, RegistryPackageTransportError> {
        let root = fs::canonicalize(&self.root).map_err(|error| {
            RegistryPackageTransportError::Unavailable(format!(
                "cannot resolve package-store root '{}': {error}",
                self.root.display()
            ))
        })?;
        let canonical = fs::canonicalize(path).map_err(|error| {
            RegistryPackageTransportError::Unavailable(format!(
                "cannot resolve package-store path '{}': {error}",
                path.display()
            ))
        })?;
        if !canonical.starts_with(&root) {
            return Err(RegistryPackageTransportError::InvalidResponse(format!(
                "package-store path '{}' escapes configured root",
                path.display()
            )));
        }
        Ok(canonical)
    }

    fn read_envelope(
        &self,
        package: &str,
        version: &str,
    ) -> Result<RegistryPackageIndexEnvelope, RegistryPackageTransportError> {
        let path = self.canonical_path_within_root(&self.package_index_path(package, version)?)?;
        let mut file = fs::File::open(&path).map_err(|error| {
            RegistryPackageTransportError::Unavailable(format!(
                "cannot open package index '{package}@{version}': {error}"
            ))
        })?;
        let mut bytes = Vec::new();
        let max_bytes = MAX_PACKAGE_INDEX_BYTES.saturating_add(1);
        file.by_ref()
            .take(max_bytes as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| {
                RegistryPackageTransportError::Unavailable(format!(
                    "cannot read package index '{package}@{version}': {error}"
                ))
            })?;
        if bytes.len() > MAX_PACKAGE_INDEX_BYTES {
            return Err(RegistryPackageTransportError::InvalidResponse(format!(
                "package index '{package}@{version}' exceeds {MAX_PACKAGE_INDEX_BYTES} bytes"
            )));
        }
        let json = String::from_utf8(bytes).map_err(|error| {
            RegistryPackageTransportError::InvalidResponse(format!(
                "package index '{package}@{version}' is not UTF-8: {error}"
            ))
        })?;
        let envelope = RegistryPackageIndexEnvelope::from_json(&json).map_err(|error| {
            RegistryPackageTransportError::InvalidResponse(format!(
                "package index '{package}@{version}' is invalid: {error}"
            ))
        })?;
        if envelope.index.resolve(package, version).is_none() {
            return Err(RegistryPackageTransportError::InvalidResponse(format!(
                "package index '{package}@{version}' does not contain the requested package"
            )));
        }
        Ok(envelope)
    }
}

impl RegistryPackageTransport for FilesystemPackageRegistry {
    fn fetch_package_index(
        &mut self,
        package: &str,
        version: &str,
    ) -> Result<RegistryPackageIndexEnvelope, RegistryPackageTransportError> {
        self.read_envelope(package, version)
    }
}

impl RegistryPackageDiscovery for FilesystemPackageRegistry {
    fn discover_package_versions(
        &mut self,
        package: &str,
    ) -> Result<Vec<String>, RegistryPackageDiscoveryError> {
        let package_directory = self
            .package_directory(package)
            .map_err(|error| match error {
                RegistryPackageTransportError::InvalidRequest(message) => {
                    RegistryPackageDiscoveryError::InvalidRequest(message)
                }
                RegistryPackageTransportError::Unavailable(message)
                | RegistryPackageTransportError::InvalidResponse(message) => {
                    RegistryPackageDiscoveryError::Unavailable(message)
                }
            })?;
        let directory = self
            .canonical_path_within_root(&package_directory)
            .map_err(|error| match error {
                RegistryPackageTransportError::InvalidRequest(message) => {
                    RegistryPackageDiscoveryError::InvalidRequest(message)
                }
                RegistryPackageTransportError::Unavailable(message)
                | RegistryPackageTransportError::InvalidResponse(message) => {
                    RegistryPackageDiscoveryError::Unavailable(message)
                }
            })?;
        let entries = fs::read_dir(&directory).map_err(|error| {
            RegistryPackageDiscoveryError::Unavailable(format!(
                "cannot read package directory '{}': {error}",
                directory.display()
            ))
        })?;
        let mut versions = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                RegistryPackageDiscoveryError::Unavailable(format!(
                    "cannot enumerate package directory '{}': {error}",
                    directory.display()
                ))
            })?;
            let file_type = entry.file_type().map_err(|error| {
                RegistryPackageDiscoveryError::Unavailable(format!(
                    "cannot inspect package entry '{}': {error}",
                    entry.path().display()
                ))
            })?;
            if !file_type.is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let Some(version) = path.file_stem().and_then(|stem| stem.to_str()) else {
                return Err(RegistryPackageDiscoveryError::InvalidResponse(
                    "package index filename is not valid UTF-8".into(),
                ));
            };
            validate_package_version(version).map_err(|error| {
                RegistryPackageDiscoveryError::InvalidResponse(format!(
                    "package '{package}' contains invalid version file '{version}.json': {error}"
                ))
            })?;
            // Discovery is metadata only, but reject stale/mismatched files so
            // a later exact fetch cannot select an index for another package.
            self.read_envelope(package, version).map_err(|error| {
                RegistryPackageDiscoveryError::InvalidResponse(error.to_string())
            })?;
            versions.push(version.to_owned());
        }
        versions.sort_by(|left, right| compare_versions(left, right));
        versions.dedup();
        Ok(versions)
    }
}

/// HTTP(S)-backed exact-version package-index transport for native hosts.
///
/// The compiler never discovers versions or downloads module sources here: it
/// fetches only the requested signed index envelope. Callers still apply their
/// pinned `RegistrySignaturePolicy` before mutating an import resolver. The
/// URL is assembled only from validated package/version path segments.
#[cfg(not(target_arch = "wasm32"))]
pub struct HttpPackageTransport {
    base_url: String,
    agent: ureq::Agent,
}

#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Debug for HttpPackageTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HttpPackageTransport")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl HttpPackageTransport {
    /// Construct an HTTP(S) transport with the default request timeout.
    pub fn new(base_url: impl Into<String>) -> Result<Self, RegistryPackageTransportError> {
        Self::with_timeout(base_url, std::time::Duration::from_secs(30))
    }

    /// Construct a transport with an explicit bounded request timeout.
    pub fn with_timeout(
        base_url: impl Into<String>,
        timeout: std::time::Duration,
    ) -> Result<Self, RegistryPackageTransportError> {
        let base_url = base_url.into().trim_end_matches('/').to_owned();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://"))
            || base_url.chars().any(char::is_whitespace)
            || base_url.contains('?')
            || base_url.contains('#')
            || base_url.len() <= "http://".len()
        {
            return Err(RegistryPackageTransportError::InvalidRequest(
                "package transport base URL must be an absolute HTTP(S) URL without query or fragment".into(),
            ));
        }
        if timeout.is_zero() {
            return Err(RegistryPackageTransportError::InvalidRequest(
                "package transport timeout must be positive".into(),
            ));
        }
        let agent = ureq::AgentBuilder::new().timeout(timeout).build();
        Ok(Self { base_url, agent })
    }

    /// Return the normalized endpoint base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RegistryPackageTransport for HttpPackageTransport {
    fn fetch_package_index(
        &mut self,
        package: &str,
        version: &str,
    ) -> Result<RegistryPackageIndexEnvelope, RegistryPackageTransportError> {
        validate_package_selector(package, version)
            .map_err(|error| RegistryPackageTransportError::InvalidRequest(error.to_string()))?;
        let url = format!("{}/{package}/{version}.json", self.base_url);
        let response = self.agent.get(&url).call().map_err(|error| match error {
            ureq::Error::Status(status, _) => RegistryPackageTransportError::InvalidResponse(
                format!("package index request returned HTTP status {status}"),
            ),
            ureq::Error::Transport(error) => RegistryPackageTransportError::Unavailable(format!(
                "package index request failed: {error}"
            )),
        })?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take(MAX_PACKAGE_INDEX_BYTES.saturating_add(1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| {
                RegistryPackageTransportError::Unavailable(format!(
                    "cannot read HTTP package index '{package}@{version}': {error}"
                ))
            })?;
        if bytes.len() > MAX_PACKAGE_INDEX_BYTES {
            return Err(RegistryPackageTransportError::InvalidResponse(format!(
                "HTTP package index '{package}@{version}' exceeds {MAX_PACKAGE_INDEX_BYTES} bytes"
            )));
        }
        let json = String::from_utf8(bytes).map_err(|error| {
            RegistryPackageTransportError::InvalidResponse(format!(
                "HTTP package index '{package}@{version}' is not UTF-8: {error}"
            ))
        })?;
        let envelope = RegistryPackageIndexEnvelope::from_json(&json).map_err(|error| {
            RegistryPackageTransportError::InvalidResponse(format!(
                "HTTP package index '{package}@{version}' is invalid: {error}"
            ))
        })?;
        if envelope.index.resolve(package, version).is_none() {
            return Err(RegistryPackageTransportError::InvalidResponse(format!(
                "HTTP package index '{package}@{version}' does not contain the requested package"
            )));
        }
        Ok(envelope)
    }
}

#[derive(Debug, Clone, Serialize)]
struct PackageDigest<'a> {
    name: &'a str,
    version: &'a str,
    modules: &'a [RegistryPackageModule],
    dependencies: &'a [RegistryPackageDependency],
}

impl RegistryPackageManifest {
    /// Compute the hash that must be stored in `manifest_hash`.
    pub fn canonical_hash(&self) -> Result<StructuralHash, CompileError> {
        let mut canonical = self.clone();
        canonical
            .modules
            .sort_by(|left, right| left.path.cmp(&right.path));
        canonical.dependencies.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| compare_versions(&left.version, &right.version))
        });
        package_manifest_hash(&canonical)
    }
}

impl RegistryPackageIndex {
    /// Parse, canonicalize, and validate a package index JSON envelope.
    pub fn from_json(json: &str) -> Result<Self, CompileError> {
        if json.len() > MAX_PACKAGE_INDEX_BYTES {
            return Err(CompileError::Unsupported(format!(
                "registry package index exceeds {MAX_PACKAGE_INDEX_BYTES} bytes"
            )));
        }
        let index: Self = serde_json::from_str(json).map_err(|error| {
            CompileError::Parse(format!("invalid registry package index: {error}"))
        })?;
        index.validate_and_canonicalize()
    }

    /// Validate and return deterministic package/key/module ordering.
    pub fn validate_and_canonicalize(mut self) -> Result<Self, CompileError> {
        if self.schema_version != REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION {
            return Err(CompileError::Unsupported(format!(
                "unsupported registry package index schema {}; expected {}",
                self.schema_version, REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION
            )));
        }
        if self.packages.is_empty() {
            return Err(CompileError::Unsupported(
                "registry package index must contain at least one package".into(),
            ));
        }
        if self.packages.len() > MAX_PACKAGES {
            return Err(CompileError::Unsupported(format!(
                "registry package index contains {} packages; maximum is {MAX_PACKAGES}",
                self.packages.len()
            )));
        }

        for package in &mut self.packages {
            validate_package_name(&package.name)?;
            validate_package_version(&package.version)?;
            if package.modules.is_empty() {
                return Err(CompileError::Unsupported(format!(
                    "package '{}@{}' must contain at least one module",
                    package.name, package.version
                )));
            }
            if package.modules.len() > MAX_MODULES_PER_PACKAGE {
                return Err(CompileError::Unsupported(format!(
                    "package '{}@{}' contains too many modules",
                    package.name, package.version
                )));
            }
            if package.dependencies.len() > MAX_DEPENDENCIES_PER_PACKAGE {
                return Err(CompileError::Unsupported(format!(
                    "package '{}@{}' contains too many dependencies",
                    package.name, package.version
                )));
            }
            for module in &package.modules {
                validate_module_path(&package.name, &package.version, &module.path)?;
                if module.source.is_empty() {
                    return Err(CompileError::Unsupported(format!(
                        "registry module '{}' has empty source",
                        module.path
                    )));
                }
            }
            for dependency in &package.dependencies {
                validate_package_name(&dependency.name)?;
                validate_package_version(&dependency.version)?;
                if dependency.name == package.name && dependency.version == package.version {
                    return Err(CompileError::Unsupported(format!(
                        "package '{}@{}' cannot depend on itself",
                        package.name, package.version
                    )));
                }
                if let Some(hash) = &dependency.manifest_hash {
                    validate_hash(hash, "dependency manifest_hash")?;
                }
            }
            package
                .modules
                .sort_by(|left, right| left.path.cmp(&right.path));
            package.dependencies.sort_by(|left, right| {
                left.name
                    .cmp(&right.name)
                    .then_with(|| compare_versions(&left.version, &right.version))
            });
            if package
                .modules
                .windows(2)
                .any(|window| window[0].path == window[1].path)
            {
                return Err(CompileError::Unsupported(format!(
                    "package '{}@{}' contains duplicate module paths",
                    package.name, package.version
                )));
            }
            if package.dependencies.windows(2).any(|window| {
                window[0].name == window[1].name && window[0].version == window[1].version
            }) {
                return Err(CompileError::Unsupported(format!(
                    "package '{}@{}' contains duplicate dependencies",
                    package.name, package.version
                )));
            }
        }

        self.packages.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| compare_versions(&left.version, &right.version))
        });

        let mut package_keys = BTreeSet::new();
        let mut module_paths = BTreeSet::new();
        let mut manifests = HashMap::new();
        for package in &self.packages {
            let key = package_key(&package.name, &package.version);
            if !package_keys.insert(key.clone()) {
                return Err(CompileError::Unsupported(format!(
                    "duplicate package '{}@{}'",
                    package.name, package.version
                )));
            }
            for module in &package.modules {
                if !module_paths.insert(module.path.clone()) {
                    return Err(CompileError::Unsupported(format!(
                        "duplicate registry module path '{}'",
                        module.path
                    )));
                }
            }
            let expected = package_manifest_hash(package)?;
            if package.manifest_hash != expected.to_string() {
                return Err(CompileError::Unsupported(format!(
                    "package '{}@{}' manifest hash does not match canonical contents",
                    package.name, package.version
                )));
            }
            manifests.insert(key, package.manifest_hash.clone());
        }

        let graph = self
            .packages
            .iter()
            .map(|package| {
                (
                    package_key(&package.name, &package.version),
                    package
                        .dependencies
                        .iter()
                        .map(|dependency| package_key(&dependency.name, &dependency.version))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for package in &self.packages {
            for dependency in &package.dependencies {
                let key = package_key(&dependency.name, &dependency.version);
                let Some(dependency_hash) = manifests.get(&key) else {
                    return Err(CompileError::Unsupported(format!(
                        "package '{}@{}' depends on missing package '{key}'",
                        package.name, package.version
                    )));
                };
                if let Some(expected) = &dependency.manifest_hash {
                    if expected != dependency_hash {
                        return Err(CompileError::Unsupported(format!(
                            "dependency '{}@{}' manifest hash pin does not match index",
                            dependency.name, dependency.version
                        )));
                    }
                }
            }
        }
        ensure_acyclic(&graph)?;
        Ok(self)
    }

    /// Serialize the canonical package index.
    pub fn to_json(&self) -> Result<String, CompileError> {
        let canonical = self.clone().validate_and_canonicalize()?;
        serde_json::to_string(&canonical).map_err(|error| {
            CompileError::Internal(format!("failed to serialize package index: {error}"))
        })
    }

    /// Return the exact package selected by name and version, if present.
    pub fn resolve(&self, name: &str, version: &str) -> Option<&RegistryPackageManifest> {
        self.packages
            .iter()
            .find(|package| package.name == name && package.version == version)
    }

    /// Return the transitive dependency closure for one package in
    /// dependency-first order.
    ///
    /// The returned `(name, version)` pairs are canonical and deterministic:
    /// dependencies are visited in the same name/version order used by the
    /// package-index canonicalizer, and each package appears exactly once.
    /// Validation is repeated on the receiver so callers cannot accidentally
    /// derive an order from an untrusted or non-canonical index.
    pub fn dependency_order(
        &self,
        name: &str,
        version: &str,
    ) -> Result<Vec<(String, String)>, CompileError> {
        let canonical = self.clone().validate_and_canonicalize()?;
        let packages = canonical
            .packages
            .iter()
            .map(|package| ((package.name.clone(), package.version.clone()), package))
            .collect::<BTreeMap<_, _>>();
        let root = (name.to_owned(), version.to_owned());
        if !packages.contains_key(&root) {
            return Err(CompileError::Unsupported(format!(
                "package dependency root '{}@{}' is missing from the index",
                name, version
            )));
        }

        fn visit(
            key: &(String, String),
            packages: &BTreeMap<(String, String), &RegistryPackageManifest>,
            visiting: &mut BTreeSet<(String, String)>,
            visited: &mut BTreeSet<(String, String)>,
            order: &mut Vec<(String, String)>,
        ) -> Result<(), CompileError> {
            if visited.contains(key) {
                return Ok(());
            }
            if !visiting.insert(key.clone()) {
                return Err(CompileError::Unsupported(format!(
                    "package dependency cycle reaches '{}@{}'",
                    key.0, key.1
                )));
            }
            let package = packages.get(key).ok_or_else(|| {
                CompileError::Unsupported(format!(
                    "package dependency '{}@{}' is missing from the index",
                    key.0, key.1
                ))
            })?;
            for dependency in &package.dependencies {
                let dependency_key = (dependency.name.clone(), dependency.version.clone());
                visit(&dependency_key, packages, visiting, visited, order)?;
            }
            visiting.remove(key);
            visited.insert(key.clone());
            order.push(key.clone());
            Ok(())
        }

        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        let mut order = Vec::new();
        visit(&root, &packages, &mut visiting, &mut visited, &mut order)?;
        Ok(order)
    }

    /// Return an order-independent structural identity for this index.
    pub fn index_hash(&self) -> Result<StructuralHash, CompileError> {
        let canonical = self.clone().validate_and_canonicalize()?;
        StructuralHash::of(&canonical).map_err(|error| {
            CompileError::Internal(format!("failed to hash package index: {error}"))
        })
    }

    /// Build the domain-separated bytes covered by a package-index signature.
    pub fn signing_payload(&self, key_id: &str) -> Result<Vec<u8>, CompileError> {
        if key_id.trim().is_empty() {
            return Err(CompileError::Unsupported(
                "package-index signature key ID must not be empty".into(),
            ));
        }
        let index_hash = self.index_hash()?;
        Ok(format!(
            "sonium-meshlang-package-index-v1\n{}\n{}\n{}",
            REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION, key_id, index_hash
        )
        .into_bytes())
    }

    /// Verify a detached signature over this exact package index.
    pub fn verify_signature(
        &self,
        signature: &RegistryPackageIndexSignature,
    ) -> Result<StructuralHash, CompileError> {
        if signature.algorithm != "ed25519" {
            return Err(CompileError::Unsupported(
                "package-index signature algorithm must be ed25519".into(),
            ));
        }
        if signature.key_id.trim().is_empty() {
            return Err(CompileError::Unsupported(
                "package-index signature key ID must not be empty".into(),
            ));
        }
        let public_key = decode_signature_hex::<32>(&signature.public_key, "public_key")?;
        let signature_bytes = decode_signature_hex::<64>(&signature.signature, "signature")?;
        let payload = self.signing_payload(&signature.key_id)?;
        let verification_key = VerificationKey::try_from(VerificationKeyBytes::from(public_key))
            .map_err(|error| {
                CompileError::Unsupported(format!("invalid package-index public key: {error:?}"))
            })?;
        let detached = Signature::from_bytes(&signature_bytes);
        verification_key.verify(&detached, &payload).map_err(|_| {
            CompileError::Unsupported("package-index signature verification failed".into())
        })?;
        self.index_hash()
    }

    /// Flatten modules in canonical package/path order for the existing
    /// source-registry resolver and signature API.
    pub fn modules(&self) -> Result<Vec<RegistryPackageModule>, CompileError> {
        let canonical = self.clone().validate_and_canonicalize()?;
        Ok(canonical
            .packages
            .into_iter()
            .flat_map(|package| package.modules)
            .collect())
    }

    /// Flatten modules under an explicit migration policy.
    ///
    /// The default [`RegistryPackageMigrationPolicy::Reject`] keeps all exact
    /// package versions addressable and never aliases an old import. An
    /// [`RegistryPackageMigrationPolicy::AllowDeclared`] plan removes the
    /// source-version modules from the registration set and returns aliases
    /// separately, so an old exact import resolves only to its declared new
    /// module.
    pub fn modules_with_migration_policy(
        &self,
        policy: &RegistryPackageMigrationPolicy,
    ) -> Result<RegistryPackageModulesWithAliases, CompileError> {
        let canonical = self.clone().validate_and_canonicalize()?;
        let (skip_package, aliases) = match policy {
            RegistryPackageMigrationPolicy::Reject => (None, Vec::new()),
            RegistryPackageMigrationPolicy::AllowDeclared(plan) => {
                let aliases = plan.validate_against(&canonical)?;
                (
                    Some(package_key(&plan.package, &plan.from_version)),
                    aliases,
                )
            }
        };
        let modules = canonical
            .packages
            .into_iter()
            .filter(|package| {
                skip_package.as_deref()
                    != Some(package_key(&package.name, &package.version).as_str())
            })
            .flat_map(|package| package.modules)
            .collect();
        Ok((modules, aliases))
    }
}

impl RegistryPackageIndexEnvelope {
    /// Parse and authenticate a signed package-index envelope.
    pub fn from_json(json: &str) -> Result<Self, CompileError> {
        if json.len() > MAX_PACKAGE_INDEX_BYTES {
            return Err(CompileError::Unsupported(format!(
                "signed registry package index exceeds {MAX_PACKAGE_INDEX_BYTES} bytes"
            )));
        }
        let mut envelope: Self = serde_json::from_str(json).map_err(|error| {
            CompileError::Parse(format!("invalid signed registry package index: {error}"))
        })?;
        envelope.index = envelope.index.validate_and_canonicalize()?;
        envelope.index.verify_signature(&envelope.signature)?;
        Ok(envelope)
    }

    /// Serialize an authenticated package-index envelope.
    pub fn to_json(&self) -> Result<String, CompileError> {
        let canonical_index = self.index.clone().validate_and_canonicalize()?;
        canonical_index.verify_signature(&self.signature)?;
        let canonical = Self {
            index: canonical_index,
            signature: self.signature.clone(),
        };
        serde_json::to_string(&canonical).map_err(|error| {
            CompileError::Internal(format!("failed to serialize signed package index: {error}"))
        })
    }

    /// Verify this envelope against a host-owned package-index trust policy.
    ///
    /// [`Self::from_json`] authenticates the detached signature and canonical
    /// contents, while this method additionally enforces the caller's pinned
    /// key set. Keeping the two checks separate lets callers parse/cache an
    /// envelope before selecting the trust policy that is active for a host.
    pub fn verify_with_policy(
        &self,
        policy: &crate::RegistrySignaturePolicy,
    ) -> Result<StructuralHash, crate::RegistrySignatureError> {
        policy.verify_package_index(&self.index, &self.signature)
    }
}

fn package_key(name: &str, version: &str) -> String {
    format!("{name}@{version}")
}

pub(crate) fn validate_package_name(name: &str) -> Result<(), CompileError> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(CompileError::Unsupported(
            "package name must not be empty".into(),
        ));
    };
    if !first.is_ascii_lowercase()
        || !chars.all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || character == '-'
                || character == '_'
        })
    {
        return Err(CompileError::Unsupported(format!(
            "invalid package name '{name}'; use lowercase ASCII package identifiers"
        )));
    }
    Ok(())
}

fn validate_package_version(version: &str) -> Result<(), CompileError> {
    if version.is_empty()
        || version.split('.').any(|part| {
            part.is_empty()
                || (part.len() > 1 && part.starts_with('0'))
                || !part.chars().all(|character| character.is_ascii_digit())
        })
    {
        return Err(CompileError::Unsupported(format!(
            "invalid package version '{version}'; expected exact numeric dotted version"
        )));
    }
    Ok(())
}

/// Validate the exact selector accepted by host-owned package transports.
///
/// Keeping this check beside the package-index grammar means transports never
/// receive selectors that could not be present in a valid index. Callers can
/// therefore reject malformed requests before doing any I/O or cache lookup.
pub(crate) fn validate_package_selector(package: &str, version: &str) -> Result<(), CompileError> {
    validate_package_name(package)?;
    validate_package_version(version)
}

fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    let left = left.split('.').collect::<Vec<_>>();
    let right = right.split('.').collect::<Vec<_>>();
    for (left_part, right_part) in left.iter().zip(&right) {
        let ordering = left_part
            .len()
            .cmp(&right_part.len())
            .then_with(|| left_part.as_bytes().cmp(right_part.as_bytes()));
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

/// Select one canonical exact version from host discovery candidates.
pub(crate) fn select_discovered_package_version(
    package: &str,
    policy: &RegistryPackageVersionPolicy,
    discovered: &[String],
) -> Result<String, CompileError> {
    validate_package_name(package)?;
    policy.validate()?;
    if discovered.is_empty() {
        return Err(CompileError::Unsupported(format!(
            "package discovery returned no versions for '{package}'"
        )));
    }

    let mut versions = discovered.to_vec();
    for version in &versions {
        validate_package_version(version)?;
    }
    versions.sort_by(|left, right| compare_versions(left, right));
    versions.dedup();

    let selected = match policy {
        RegistryPackageVersionPolicy::Exact(requested) => {
            versions.into_iter().find(|version| version == requested)
        }
        RegistryPackageVersionPolicy::HighestAtMost(bound) => versions
            .into_iter()
            .rfind(|version| compare_versions(version, bound) != std::cmp::Ordering::Greater),
    };
    selected.ok_or_else(|| {
        CompileError::Unsupported(format!(
            "no discovered version for package '{package}' satisfies the requested policy"
        ))
    })
}

fn validate_module_path(
    package: &str,
    package_version: &str,
    path: &str,
) -> Result<(), CompileError> {
    if path.is_empty()
        || path.contains('\\')
        || path.contains('\0')
        || path.starts_with('/')
        || !path.ends_with(".mlg")
        || !path.starts_with(&format!("{package}/"))
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(CompileError::Unsupported(format!(
            "invalid module path '{path}' for package '{package}'"
        )));
    }
    let filename = path.rsplit('/').next().unwrap_or_default();
    let stem = filename.strip_suffix(".mlg").unwrap_or_default();
    if stem.matches('@').count() > 1 {
        return Err(CompileError::Unsupported(format!(
            "module path '{path}' contains multiple version selectors"
        )));
    }
    if let Some((module_name, version)) = stem.rsplit_once('@') {
        if module_name.is_empty() {
            return Err(CompileError::Unsupported(format!(
                "module path '{path}' has an empty module name"
            )));
        }
        validate_package_version(version)?;
        if version != package_version {
            return Err(CompileError::Unsupported(format!(
                "module path '{path}' selects package version '{version}', expected '{package_version}'"
            )));
        }
    }
    Ok(())
}

fn validate_hash(value: &str, field: &str) -> Result<(), CompileError> {
    if value.len() != 64 || !value.chars().all(|character| character.is_ascii_hexdigit()) {
        return Err(CompileError::Unsupported(format!(
            "{field} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn decode_signature_hex<const N: usize>(value: &str, field: &str) -> Result<[u8; N], CompileError> {
    if value.len() != N * 2 {
        return Err(CompileError::Unsupported(format!(
            "package-index {field} must contain exactly {} hexadecimal characters",
            N * 2
        )));
    }
    let mut bytes = [0_u8; N];
    for (index, chunk) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let high = signature_hex_nibble(chunk[0]).ok_or_else(|| {
            CompileError::Unsupported(format!("package-index {field} contains non-hex data"))
        })?;
        let low = signature_hex_nibble(chunk[1]).ok_or_else(|| {
            CompileError::Unsupported(format!("package-index {field} contains non-hex data"))
        })?;
        bytes[index] = (high << 4) | low;
    }
    Ok(bytes)
}

fn signature_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn package_manifest_hash(
    package: &RegistryPackageManifest,
) -> Result<StructuralHash, CompileError> {
    StructuralHash::of(&PackageDigest {
        name: &package.name,
        version: &package.version,
        modules: &package.modules,
        dependencies: &package.dependencies,
    })
    .map_err(|error| CompileError::Internal(format!("failed to hash package manifest: {error}")))
}

fn ensure_acyclic(graph: &BTreeMap<String, Vec<String>>) -> Result<(), CompileError> {
    fn visit(
        key: &str,
        graph: &BTreeMap<String, Vec<String>>,
        active: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) -> Result<(), CompileError> {
        if visited.contains(key) {
            return Ok(());
        }
        if !active.insert(key.to_owned()) {
            return Err(CompileError::Unsupported(format!(
                "registry package dependency cycle includes '{key}'"
            )));
        }
        if let Some(dependencies) = graph.get(key) {
            for dependency in dependencies {
                visit(dependency, graph, active, visited)?;
            }
        }
        active.remove(key);
        visited.insert(key.to_owned());
        Ok(())
    }

    let mut active = HashSet::new();
    let mut visited = HashSet::new();
    for key in graph.keys() {
        visit(key, graph, &mut active, &mut visited)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ed25519_zebra::{SigningKey, VerificationKeyBytes};

    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn package(
        name: &str,
        version: &str,
        path: &str,
        dependencies: Vec<RegistryPackageDependency>,
    ) -> RegistryPackageManifest {
        let mut package = RegistryPackageManifest {
            name: name.into(),
            version: version.into(),
            modules: vec![RegistryPackageModule {
                path: path.into(),
                source: format!("pub const VERSION = {version}\n"),
            }],
            dependencies,
            manifest_hash: String::new(),
        };
        package.manifest_hash = package_manifest_hash(&package).unwrap().to_string();
        package
    }

    #[test]
    fn package_index_canonicalizes_order_and_resolves_exact_versions() {
        let base = package("base", "1.0", "base/core@1.0.mlg", Vec::new());
        let mut hash_order_a = package("hash", "1", "hash/z.mlg", Vec::new());
        hash_order_a.modules.push(RegistryPackageModule {
            path: "hash/a.mlg".into(),
            source: "a".into(),
        });
        let mut hash_order_b = hash_order_a.clone();
        hash_order_b.modules.reverse();
        assert_eq!(
            hash_order_a.canonical_hash().unwrap(),
            hash_order_b.canonical_hash().unwrap()
        );
        let app = package(
            "app",
            "2",
            "app/main@2.mlg",
            vec![RegistryPackageDependency {
                name: "base".into(),
                version: "1.0".into(),
                manifest_hash: Some(base.manifest_hash.clone()),
            }],
        );
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![app.clone(), base.clone()],
        }
        .validate_and_canonicalize()
        .unwrap();
        assert_eq!(index.packages[0].name, "app");
        assert_eq!(index.resolve("base", "1.0"), Some(&base));
        assert!(index.resolve("base", "1").is_none());
        assert_eq!(index.modules().unwrap().len(), 2);
        let reversed = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![base, app],
        }
        .validate_and_canonicalize()
        .unwrap();
        assert_eq!(index.index_hash().unwrap(), reversed.index_hash().unwrap());
        assert_eq!(index.to_json().unwrap(), reversed.to_json().unwrap());
    }

    #[test]
    fn package_index_dependency_order_is_stable_and_dependency_first() {
        let base = package("base", "1", "base/main.mlg", Vec::new());
        let mut shared = package("shared", "1", "shared/main.mlg", Vec::new());
        shared.dependencies.push(RegistryPackageDependency {
            name: "base".into(),
            version: "1".into(),
            manifest_hash: Some(base.manifest_hash.clone()),
        });
        shared.manifest_hash = shared.canonical_hash().unwrap().to_string();
        let mut app = package("app", "1", "app/main.mlg", Vec::new());
        app.dependencies.push(RegistryPackageDependency {
            name: "shared".into(),
            version: "1".into(),
            manifest_hash: Some(shared.manifest_hash.clone()),
        });
        app.dependencies.push(RegistryPackageDependency {
            name: "base".into(),
            version: "1".into(),
            manifest_hash: Some(base.manifest_hash.clone()),
        });
        app.manifest_hash = app.canonical_hash().unwrap().to_string();
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![app.clone(), base.clone(), shared.clone()],
        };
        assert_eq!(
            index.dependency_order("app", "1").unwrap(),
            vec![
                ("base".to_owned(), "1".to_owned()),
                ("shared".to_owned(), "1".to_owned()),
                ("app".to_owned(), "1".to_owned()),
            ]
        );
        let reversed = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![shared, app, base],
        };
        assert_eq!(
            index.dependency_order("app", "1").unwrap(),
            reversed.dependency_order("app", "1").unwrap()
        );
        let missing = index.dependency_order("missing", "1").unwrap_err();
        assert!(missing.to_string().contains("dependency root 'missing@1'"));
    }

    #[test]
    fn package_index_rejects_missing_pins_cycles_and_host_paths() {
        let missing = package(
            "app",
            "1",
            "app/main.mlg",
            vec![RegistryPackageDependency {
                name: "missing".into(),
                version: "1".into(),
                manifest_hash: None,
            }],
        );
        let error = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![missing],
        }
        .validate_and_canonicalize()
        .unwrap_err()
        .to_string();
        assert!(error.contains("missing package"), "{error}");

        let mut bad_path = package("app", "1", "app/main.mlg", Vec::new());
        bad_path.modules[0].path = "app\\main.mlg".into();
        let error = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![bad_path],
        }
        .validate_and_canonicalize()
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid module path"), "{error}");

        let mut bad_version = package("app", "1", "app/main.mlg", Vec::new());
        bad_version.modules[0].path = "app/main@beta.mlg".into();
        let error = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![bad_version],
        }
        .validate_and_canonicalize()
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid package version"), "{error}");

        let mut mismatched_version = package("app", "1", "app/main.mlg", Vec::new());
        mismatched_version.modules[0].path = "app/main@2.mlg".into();
        let error = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![mismatched_version],
        }
        .validate_and_canonicalize()
        .unwrap_err()
        .to_string();
        assert!(error.contains("expected '1'"), "{error}");

        let mut first = package("first", "1", "first/main.mlg", Vec::new());
        let mut second = package("second", "1", "second/main.mlg", Vec::new());
        first.dependencies.push(RegistryPackageDependency {
            name: "second".into(),
            version: "1".into(),
            manifest_hash: None,
        });
        second.dependencies.push(RegistryPackageDependency {
            name: "first".into(),
            version: "1".into(),
            manifest_hash: None,
        });
        first.manifest_hash = package_manifest_hash(&first).unwrap().to_string();
        second.manifest_hash = package_manifest_hash(&second).unwrap().to_string();
        let error = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![first, second],
        }
        .validate_and_canonicalize()
        .unwrap_err()
        .to_string();
        assert!(error.contains("cycle"), "{error}");
    }

    #[test]
    fn package_index_json_round_trip_rejects_schema_and_hash_drift() {
        let package = package("demo", "1.2", "demo/main@1.2.mlg", Vec::new());
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![package],
        };
        let json = index.to_json().unwrap();
        let parsed = RegistryPackageIndex::from_json(&json).unwrap();
        assert_eq!(parsed.to_json().unwrap(), json);

        let unsupported = json.replace("\"schema_version\":1", "\"schema_version\":9");
        assert!(RegistryPackageIndex::from_json(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("unsupported registry package index schema"));
        let tampered = json.replace("VERSION = 1.2", "VERSION = 9.9");
        assert!(RegistryPackageIndex::from_json(&tampered)
            .unwrap_err()
            .to_string()
            .contains("manifest hash"));
    }

    #[test]
    fn package_index_envelope_canonicalizes_order_before_serialization() {
        let mut signed = package("signed", "1", "signed/z.mlg", Vec::new());
        signed.modules.push(RegistryPackageModule {
            path: "signed/a.mlg".into(),
            source: "a".into(),
        });
        signed.manifest_hash = signed.canonical_hash().unwrap().to_string();
        let other = package("other", "1", "other/main.mlg", Vec::new());
        let canonical_index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![signed.clone(), other.clone()],
        }
        .validate_and_canonicalize()
        .unwrap();

        let signing_key = SigningKey::from([8_u8; 32]);
        let key_id = "qa-package-order-key-v1";
        let detached = signing_key.sign(&canonical_index.signing_payload(key_id).unwrap());
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        let signature = RegistryPackageIndexSignature {
            algorithm: "ed25519".into(),
            key_id: key_id.into(),
            public_key: hex(&public_key),
            signature: hex(&detached.to_bytes()),
        };

        let mut scrambled = signed;
        scrambled.modules.reverse();
        let envelope = RegistryPackageIndexEnvelope {
            index: RegistryPackageIndex {
                schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
                packages: vec![other, scrambled],
            },
            signature: signature.clone(),
        };
        let canonical_envelope = RegistryPackageIndexEnvelope {
            index: canonical_index.clone(),
            signature,
        };
        let json = envelope.to_json().unwrap();
        assert_eq!(json, canonical_envelope.to_json().unwrap());

        let parsed = RegistryPackageIndexEnvelope::from_json(&json).unwrap();
        assert_eq!(parsed.index, canonical_index);
        assert_eq!(parsed.to_json().unwrap(), json);
    }

    #[test]
    fn package_index_signature_binds_canonical_index_and_key_id() {
        let package = package("signed", "1", "signed/main@1.mlg", Vec::new());
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![package],
        };
        let signing_key = SigningKey::from([7_u8; 32]);
        let key_id = "qa-package-key-v1";
        let payload = index.signing_payload(key_id).unwrap();
        let detached = signing_key.sign(&payload);
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        let signature = RegistryPackageIndexSignature {
            algorithm: "ed25519".into(),
            key_id: key_id.into(),
            public_key: hex(&public_key),
            signature: hex(&detached.to_bytes()),
        };
        assert_eq!(
            index.verify_signature(&signature).unwrap(),
            index.index_hash().unwrap()
        );
        let envelope = RegistryPackageIndexEnvelope {
            index: index.clone(),
            signature: signature.clone(),
        };
        let json = envelope.to_json().unwrap();
        assert_eq!(
            RegistryPackageIndexEnvelope::from_json(&json)
                .unwrap()
                .index
                .index_hash()
                .unwrap(),
            index.index_hash().unwrap()
        );

        let mut tampered = index.clone();
        tampered.packages[0].modules[0].source.push_str("tampered");
        assert!(tampered.verify_signature(&signature).is_err());

        let mut wrong_key_id = signature.clone();
        wrong_key_id.key_id = "other-key".into();
        assert!(index.verify_signature(&wrong_key_id).is_err());

        let mut malformed = signature;
        malformed.signature = "00".into();
        let error = index.verify_signature(&malformed).unwrap_err().to_string();
        assert!(error.contains("exactly 128"), "{error}");
    }

    #[test]
    fn package_index_signature_requires_pinned_key_and_detects_tampering() {
        let package = package("signed", "1", "signed/main@1.mlg", Vec::new());
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![package],
        };
        let signing_key = SigningKey::from([9_u8; 32]);
        let key_id = "package-key-v1";
        let signature_bytes = signing_key.sign(&index.signing_payload(key_id).unwrap());
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        let signature = RegistryPackageIndexSignature {
            algorithm: "ed25519".into(),
            key_id: key_id.into(),
            public_key: hex(&public_key),
            signature: hex(&signature_bytes.to_bytes()),
        };
        let policy = crate::RegistrySignaturePolicy {
            trusted_keys: vec![crate::RegistryTrustAnchor {
                key_id: key_id.into(),
                public_key: signature.public_key.clone(),
            }],
        };

        assert_eq!(
            policy.verify_package_index(&index, &signature).unwrap(),
            index.index_hash().unwrap()
        );

        let mut untrusted = signature.clone();
        untrusted.key_id = "other-key".into();
        assert!(matches!(
            policy.verify_package_index(&index, &untrusted),
            Err(crate::RegistrySignatureError::UntrustedKeyId(id)) if id == "other-key"
        ));

        let mut wrong_pin = signature.clone();
        wrong_pin.public_key = "11".repeat(32);
        assert!(matches!(
            policy.verify_package_index(&index, &wrong_pin),
            Err(crate::RegistrySignatureError::PublicKeyMismatch { .. })
        ));

        let mut tampered = index.clone();
        tampered.packages[0].modules[0].source.push_str("tampered");
        assert!(policy.verify_package_index(&tampered, &signature).is_err());
    }

    #[test]
    fn discovered_version_selection_is_deterministic_and_explicit() {
        let candidates = vec!["1".into(), "1.0".into(), "2".into(), "1".into()];
        assert_eq!(
            select_discovered_package_version(
                "demo",
                &RegistryPackageVersionPolicy::Exact("1".into()),
                &candidates,
            )
            .unwrap(),
            "1"
        );
        assert_eq!(
            select_discovered_package_version(
                "demo",
                &RegistryPackageVersionPolicy::HighestAtMost("1.5".into()),
                &candidates,
            )
            .unwrap(),
            "1.0"
        );
        let error = select_discovered_package_version(
            "demo",
            &RegistryPackageVersionPolicy::Exact("3".into()),
            &candidates,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no discovered version"), "{error}");
        let error = select_discovered_package_version(
            "demo",
            &RegistryPackageVersionPolicy::Exact("1".into()),
            &["01".into()],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid package version"), "{error}");
    }

    fn signed_index_json(name: &str, version: &str, key_seed: u8) -> String {
        let package = package(
            name,
            version,
            &format!("{name}/main@{version}.mlg"),
            Vec::new(),
        );
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![package],
        };
        let signing_key = SigningKey::from([key_seed; 32]);
        let key_id = format!("filesystem-key-{key_seed}");
        let signature_bytes = signing_key.sign(&index.signing_payload(&key_id).unwrap());
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        RegistryPackageIndexEnvelope {
            index,
            signature: RegistryPackageIndexSignature {
                algorithm: "ed25519".into(),
                key_id,
                public_key: hex(&public_key),
                signature: hex(&signature_bytes.to_bytes()),
            },
        }
        .to_json()
        .unwrap()
    }

    #[test]
    fn filesystem_package_registry_discovers_and_fetches_signed_exact_versions() {
        let directory = tempfile::tempdir().unwrap();
        let package_directory = directory.path().join("demo");
        std::fs::create_dir_all(&package_directory).unwrap();
        std::fs::write(
            package_directory.join("1.0.json"),
            signed_index_json("demo", "1.0", 21),
        )
        .unwrap();
        std::fs::write(
            package_directory.join("2.json"),
            signed_index_json("demo", "2", 22),
        )
        .unwrap();
        std::fs::write(package_directory.join("README.txt"), "ignored").unwrap();

        let mut registry = FilesystemPackageRegistry::new(directory.path());
        assert_eq!(
            registry.discover_package_versions("demo").unwrap(),
            vec!["1.0".to_owned(), "2".to_owned()]
        );
        let envelope = registry.fetch_package_index("demo", "1.0").unwrap();
        assert!(envelope.index.resolve("demo", "1.0").is_some());
    }

    #[test]
    fn filesystem_package_registry_rejects_traversal_invalid_and_mismatched_entries() {
        let directory = tempfile::tempdir().unwrap();
        let package_directory = directory.path().join("demo");
        std::fs::create_dir_all(&package_directory).unwrap();
        std::fs::write(
            package_directory.join("3.json"),
            signed_index_json("other", "3", 23),
        )
        .unwrap();

        let mut registry = FilesystemPackageRegistry::new(directory.path());
        let traversal = registry.fetch_package_index("../demo", "1").unwrap_err();
        assert!(matches!(
            traversal,
            RegistryPackageTransportError::InvalidRequest(_)
        ));
        let mismatch = registry.fetch_package_index("demo", "3").unwrap_err();
        assert!(matches!(
            mismatch,
            RegistryPackageTransportError::InvalidResponse(_)
        ));

        std::fs::write(package_directory.join("4.json"), "not-json").unwrap();
        let error = registry.discover_package_versions("demo").unwrap_err();
        assert!(matches!(
            error,
            RegistryPackageDiscoveryError::InvalidResponse(_)
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn http_package_transport_validates_endpoint_and_timeout_without_network_io() {
        assert!(matches!(
            HttpPackageTransport::new("file:///tmp/registry"),
            Err(RegistryPackageTransportError::InvalidRequest(_))
        ));
        assert!(matches!(
            HttpPackageTransport::new("https://registry.example/?token=secret"),
            Err(RegistryPackageTransportError::InvalidRequest(_))
        ));
        assert!(matches!(
            HttpPackageTransport::with_timeout(
                "https://registry.example",
                std::time::Duration::ZERO,
            ),
            Err(RegistryPackageTransportError::InvalidRequest(_))
        ));
        let transport = HttpPackageTransport::new("https://registry.example///").unwrap();
        assert_eq!(transport.base_url(), "https://registry.example");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "loopback sockets are unavailable in the sandbox; integration hosts exercise HTTP transport"]
    fn http_package_transport_fetches_exact_signed_index_and_rejects_status() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let body = signed_index_json("demo", "1", 24);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn({
            let body = body.clone();
            move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 2048];
                let length = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..length]);
                assert!(request.starts_with("GET /demo/1.json HTTP/1.1"));
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .unwrap();
            }
        });
        let mut transport = HttpPackageTransport::with_timeout(
            format!("http://{address}/registry/"),
            std::time::Duration::from_secs(2),
        )
        .unwrap();
        let envelope = transport.fetch_package_index("demo", "1").unwrap();
        assert!(envelope.index.resolve("demo", "1").is_some());
        server.join().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 256];
            let _ = stream.read(&mut request);
            stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        let mut transport = HttpPackageTransport::new(format!("http://{address}")).unwrap();
        let error = transport.fetch_package_index("demo", "1").unwrap_err();
        assert!(matches!(
            error,
            RegistryPackageTransportError::InvalidResponse(_)
        ));
        server.join().unwrap();
    }
}
