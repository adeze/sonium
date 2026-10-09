//! CPU-only MeshLang package-store transport and discovery reference.

use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use meshlang_compiler::{
    FilesystemPackageRegistry, REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION, RegistryPackageDiscovery,
    RegistryPackageIndex, RegistryPackageIndexEnvelope, RegistryPackageIndexSignature,
    RegistryPackageManifest, RegistryPackageModule, RegistryPackageTransport,
    RegistryPackageTransportError, RegistrySignaturePolicy, RegistryTrustAnchor,
    imports::ImportResolver,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.registry-filesystem-transport-reference.v1";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn signed_index(version: &str, signing_key: &SigningKey) -> (String, String) {
    let mut package = RegistryPackageManifest {
        name: "demo".into(),
        version: version.into(),
        modules: vec![RegistryPackageModule {
            path: format!("demo/main@{version}.mlg"),
            source: format!("pub const VERSION = {version}\n"),
        }],
        dependencies: Vec::new(),
        manifest_hash: String::new(),
    };
    package.manifest_hash = package.canonical_hash().expect("package hash").to_string();
    let index = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![package],
    };
    let key_id = "qa-filesystem-key-v1";
    let signature = signing_key.sign(&index.signing_payload(key_id).expect("signing payload"));
    let public_key: [u8; 32] = VerificationKeyBytes::from(signing_key).into();
    let envelope = RegistryPackageIndexEnvelope {
        index,
        signature: RegistryPackageIndexSignature {
            algorithm: "ed25519".into(),
            key_id: key_id.into(),
            public_key: hex(&public_key),
            signature: hex(&signature.to_bytes()),
        },
    };
    (
        envelope.to_json().expect("signed index JSON"),
        hex(&public_key),
    )
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let package_directory = directory.path().join("demo");
    std::fs::create_dir_all(&package_directory).map_err(|error| error.to_string())?;
    let signing_key = SigningKey::from([42_u8; 32]);
    let (version_one, public_key) = signed_index("1", &signing_key);
    let (version_two, _) = signed_index("2", &signing_key);
    std::fs::write(package_directory.join("1.json"), version_one)
        .map_err(|error| error.to_string())?;
    std::fs::write(package_directory.join("2.json"), version_two)
        .map_err(|error| error.to_string())?;
    std::fs::write(package_directory.join("README.txt"), "ignored")
        .map_err(|error| error.to_string())?;

    let mut registry = FilesystemPackageRegistry::new(directory.path());
    let versions = registry
        .discover_package_versions("demo")
        .map_err(|error| error.to_string())?;
    if versions != vec!["1".to_owned(), "2".to_owned()] {
        return Err(format!("filesystem discovery returned {versions:?}"));
    }
    let exact = registry
        .fetch_package_index("demo", "2")
        .map_err(|error| error.to_string())?;
    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: "qa-filesystem-key-v1".into(),
            public_key,
        }],
    };
    let expected_hash = exact
        .index
        .index_hash()
        .map_err(|error| error.to_string())?;
    let mut resolver = ImportResolver::new();
    let registered_hash = resolver
        .register_trusted_package_from_transport("demo", "2", &mut registry, &policy)
        .map_err(|error| error.to_string())?;
    if registered_hash != expected_hash {
        return Err("filesystem transport hash did not match signed index".into());
    }
    resolver
        .load_module("demo/main@2.mlg", std::path::Path::new("/virtual/main.mlg"))
        .map_err(|error| error.to_string())?;

    let traversal = registry.fetch_package_index("../demo", "2");
    if !matches!(
        traversal,
        Err(RegistryPackageTransportError::InvalidRequest(_))
    ) {
        return Err("filesystem transport accepted a traversal selector".into());
    }

    #[cfg(not(target_arch = "wasm32"))]
    let endpoint_check = meshlang_compiler::HttpPackageTransport::new("file:///not-http");
    #[cfg(not(target_arch = "wasm32"))]
    if !matches!(
        endpoint_check,
        Err(RegistryPackageTransportError::InvalidRequest(_))
    ) {
        return Err("HTTP transport accepted a non-HTTP endpoint".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("discovered_versions", versions.len() as f64)
        .metric("trusted_registration_checks", 3.0)
        .metric("selector_rejection_checks", 1.0)
        .metric("http_endpoint_checks", 1.0)
        .dimension("native filesystem package store with signed exact-version indexes")
        .input_checksum("temp-store:demo-1-and-2:meshlang-registry-transport-v1")
        .reference_checksum("ed25519-signed-index-and-resolver-hash:v1")
        .formulation("filesystem discovery plus exact signed RegistryPackageTransport")
        .tolerance("exact version ordering, trust hash, module resolution, and path rejection")
        .seed("meshlang-registry-filesystem-transport-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
