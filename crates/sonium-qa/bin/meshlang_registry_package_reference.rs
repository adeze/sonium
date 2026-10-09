//! Deterministic local MeshLang package-index/version policy reference.
//!
//! This case validates exact package resolution, canonical ordering, manifest
//! hashes, dependency pins/cycles, and portable module paths. It performs no
//! network access, Gmsh execution, or simulation workload.

use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use meshlang_compiler::{
    IncrementalCompiler, REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
    REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION, RegistryPackageDependency, RegistryPackageDiscovery,
    RegistryPackageDiscoveryError, RegistryPackageIndex, RegistryPackageIndexEnvelope,
    RegistryPackageIndexSignature, RegistryPackageManifest, RegistryPackageMigrationModule,
    RegistryPackageMigrationPlan, RegistryPackageMigrationPolicy, RegistryPackageModule,
    RegistryPackageTransport, RegistryPackageTransportError, RegistryPackageVersionPolicy,
    RegistrySignaturePolicy, RegistryTrustAnchor, imports::ImportResolver,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.registry-package-reference.v1";

fn package(name: &str, version: &str, path: &str) -> RegistryPackageManifest {
    let mut package = RegistryPackageManifest {
        name: name.into(),
        version: version.into(),
        modules: vec![RegistryPackageModule {
            path: path.into(),
            source: format!("pub const VERSION = {version}\n"),
        }],
        dependencies: Vec::new(),
        manifest_hash: String::new(),
    };
    package.manifest_hash = package
        .canonical_hash()
        .expect("canonical package hash")
        .to_string();
    package
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct FixtureTransport {
    envelope: RegistryPackageIndexEnvelope,
    requests: Vec<(String, String)>,
}

impl RegistryPackageTransport for FixtureTransport {
    fn fetch_package_index(
        &mut self,
        package: &str,
        version: &str,
    ) -> Result<RegistryPackageIndexEnvelope, RegistryPackageTransportError> {
        self.requests.push((package.into(), version.into()));
        Ok(self.envelope.clone())
    }
}

struct FixtureDiscovery {
    versions: Vec<String>,
    requests: Vec<String>,
    failure: Option<RegistryPackageDiscoveryError>,
}

impl RegistryPackageDiscovery for FixtureDiscovery {
    fn discover_package_versions(
        &mut self,
        package: &str,
    ) -> Result<Vec<String>, RegistryPackageDiscoveryError> {
        self.requests.push(package.to_owned());
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        Ok(self.versions.clone())
    }
}

fn run() -> Result<QaResult, String> {
    let base = package("base", "1.0", "base/core@1.0.mlg");
    let mut app = package("app", "2", "app/main@2.mlg");
    app.dependencies.push(RegistryPackageDependency {
        name: "base".into(),
        version: "1.0".into(),
        manifest_hash: Some(base.manifest_hash.clone()),
    });
    app.manifest_hash = app
        .canonical_hash()
        .map_err(|error| error.to_string())?
        .to_string();

    let index = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![app.clone(), base.clone()],
    }
    .validate_and_canonicalize()
    .map_err(|error| error.to_string())?;

    let migration_old = package("demo", "1", "demo/root@1.mlg");
    let migration_new = package("demo", "2", "demo/root@2.mlg");
    let migration_index = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![migration_old.clone(), migration_new.clone()],
    }
    .validate_and_canonicalize()
    .map_err(|error| error.to_string())?;
    let migration_plan = RegistryPackageMigrationPlan {
        schema_version: REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION,
        package: "demo".into(),
        from_version: "1".into(),
        to_version: "2".into(),
        from_manifest_hash: migration_old.manifest_hash.clone(),
        to_manifest_hash: migration_new.manifest_hash.clone(),
        modules: vec![RegistryPackageMigrationModule {
            from: "demo/root@1.mlg".into(),
            to: "demo/root@2.mlg".into(),
        }],
    };
    if migration_plan
        .validate_against(&migration_index)
        .map_err(|error| error.to_string())?
        != vec![("demo/root@1.mlg".to_owned(), "demo/root@2.mlg".to_owned())]
    {
        return Err("package migration alias validation drifted".into());
    }
    let mut migration_exact = ImportResolver::new();
    migration_exact
        .register_package_index(&migration_index)
        .map_err(|error| error.to_string())?;
    if migration_exact.registered_migrations() != Vec::<(String, String)>::new()
        || migration_exact.registered_modules()
            != vec!["demo/root@1.mlg".to_owned(), "demo/root@2.mlg".to_owned()]
    {
        return Err("default package migration policy changed exact registration".into());
    }
    let mut migration_alias = ImportResolver::new();
    migration_alias
        .register_package_index_with_migration_policy(
            &migration_index,
            &RegistryPackageMigrationPolicy::AllowDeclared(migration_plan.clone()),
        )
        .map_err(|error| error.to_string())?;
    if migration_alias.registered_modules() != vec!["demo/root@2.mlg".to_owned()]
        || migration_alias.registered_migrations()
            != vec![("demo/root@1.mlg".to_owned(), "demo/root@2.mlg".to_owned())]
    {
        return Err("declared package migration alias registration drifted".into());
    }
    if migration_alias
        .resolve_path("demo/root@1.mlg", std::path::Path::new("/virtual/main.mlg"))
        .map_err(|error| error.to_string())?
        .to_string_lossy()
        != "<meshlang-registry>/demo/root@2.mlg"
    {
        return Err("old package import did not resolve through declared migration".into());
    }
    let mut invalid_migration = migration_plan;
    invalid_migration.modules[0].to = "demo/missing@2.mlg".into();
    let mut invalid_migration_resolver = ImportResolver::new();
    if invalid_migration_resolver
        .register_package_index_with_migration_policy(
            &migration_index,
            &RegistryPackageMigrationPolicy::AllowDeclared(invalid_migration),
        )
        .is_ok()
        || !invalid_migration_resolver.registered_modules().is_empty()
        || !invalid_migration_resolver
            .registered_migrations()
            .is_empty()
    {
        return Err("invalid package migration mutated the resolver".into());
    }

    let mut resolver = ImportResolver::new();
    resolver
        .register_package_index(&index)
        .map_err(|error| error.to_string())?;
    if resolver.registered_modules()
        != vec!["app/main@2.mlg".to_owned(), "base/core@1.0.mlg".to_owned()]
    {
        return Err("package-index modules were not registered canonically".into());
    }
    resolver
        .load_module("app/main@2.mlg", std::path::Path::new("/virtual/main.mlg"))
        .map_err(|error| error.to_string())?;
    if index.resolve("base", "1.0") != Some(&base) || index.resolve("base", "1").is_some() {
        return Err("exact package version resolution drifted".into());
    }
    let dependency_order = index
        .dependency_order("app", "2")
        .map_err(|error| error.to_string())?;
    if dependency_order
        != vec![
            ("base".to_owned(), "1.0".to_owned()),
            ("app".to_owned(), "2".to_owned()),
        ]
    {
        return Err("dependency-first package order drifted".into());
    }
    if !index
        .dependency_order("missing", "1")
        .map_err(|error| error.to_string())
        .unwrap_err()
        .contains("dependency root 'missing@1'")
    {
        return Err("missing dependency root was not rejected".into());
    }
    if index.modules().map_err(|error| error.to_string())?.len() != 2 {
        return Err("canonical module flattening drifted".into());
    }
    let json = index.to_json().map_err(|error| error.to_string())?;
    let parsed = RegistryPackageIndex::from_json(&json).map_err(|error| error.to_string())?;
    if parsed.to_json().map_err(|error| error.to_string())? != json {
        return Err("package index JSON is not canonical".into());
    }
    let reversed = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![base, app],
    }
    .validate_and_canonicalize()
    .map_err(|error| error.to_string())?;
    if parsed.index_hash().map_err(|error| error.to_string())?
        != reversed.index_hash().map_err(|error| error.to_string())?
    {
        return Err("package index hash depends on input ordering".into());
    }

    let signing_key = SigningKey::from([7_u8; 32]);
    let key_id = "qa-package-key-v1";
    let payload = index
        .signing_payload(key_id)
        .map_err(|error| error.to_string())?;
    let detached = signing_key.sign(&payload);
    let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
    let signature = RegistryPackageIndexSignature {
        algorithm: "ed25519".into(),
        key_id: key_id.into(),
        public_key: hex(&public_key),
        signature: hex(&detached.to_bytes()),
    };
    index
        .verify_signature(&signature)
        .map_err(|error| error.to_string())?;
    let envelope = RegistryPackageIndexEnvelope {
        index: index.clone(),
        signature: signature.clone(),
    };
    let signed_json = envelope.to_json().map_err(|error| error.to_string())?;
    RegistryPackageIndexEnvelope::from_json(&signed_json).map_err(|error| error.to_string())?;

    let mut reordered_index = index.clone();
    reordered_index.packages.reverse();
    let reordered_envelope = RegistryPackageIndexEnvelope {
        index: reordered_index,
        signature: signature.clone(),
    };
    let reordered_json = reordered_envelope
        .to_json()
        .map_err(|error| error.to_string())?;
    if reordered_json != signed_json {
        return Err("signed package-index envelope ordering is not canonical".into());
    }
    if RegistryPackageIndexEnvelope::from_json(&reordered_json)
        .map_err(|error| error.to_string())?
        .index
        != index
    {
        return Err("parsed signed package-index envelope is not canonical".into());
    }

    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: key_id.into(),
            public_key: hex(&public_key),
        }],
    };
    let mut trusted_resolver = ImportResolver::new();
    if trusted_resolver
        .register_trusted_package_index(&envelope, &policy)
        .map_err(|error| error.to_string())?
        != index.index_hash().map_err(|error| error.to_string())?
    {
        return Err("trusted package-index import hash drifted".into());
    }
    trusted_resolver
        .load_module("app/main@2.mlg", std::path::Path::new("/virtual/main.mlg"))
        .map_err(|error| error.to_string())?;
    if envelope
        .verify_with_policy(&policy)
        .map_err(|error| error.to_string())?
        != index.index_hash().map_err(|error| error.to_string())?
    {
        return Err("trusted package-index hash drifted".into());
    }

    let mut transport = FixtureTransport {
        envelope: envelope.clone(),
        requests: Vec::new(),
    };
    let mut transported_resolver = ImportResolver::new();
    let transported_hash = transported_resolver
        .register_trusted_package_from_transport("app", "2", &mut transport, &policy)
        .map_err(|error| error.to_string())?;
    if transported_hash != index.index_hash().map_err(|error| error.to_string())?
        || transport.requests != vec![("app".into(), "2".into())]
    {
        return Err("exact package transport request/hash drifted".into());
    }
    transported_resolver
        .load_module("app/main@2.mlg", std::path::Path::new("/virtual/main.mlg"))
        .map_err(|error| error.to_string())?;

    let mut discovery = FixtureDiscovery {
        versions: vec!["1".into(), "2".into(), "0.9".into(), "2".into()],
        requests: Vec::new(),
        failure: None,
    };
    let mut discovered_transport = FixtureTransport {
        envelope: envelope.clone(),
        requests: Vec::new(),
    };
    let mut discovered_resolver = ImportResolver::new();
    let resolution = discovered_resolver
        .register_trusted_package_from_discovery(
            "app",
            &RegistryPackageVersionPolicy::HighestAtMost("2.0".into()),
            &mut discovery,
            &mut discovered_transport,
            &policy,
        )
        .map_err(|error| error.to_string())?;
    if resolution.package != "app"
        || resolution.version != "2"
        || resolution.index_hash != index.index_hash().map_err(|error| error.to_string())?
        || discovery.requests != vec!["app"]
        || discovered_transport.requests != vec![("app".into(), "2".into())]
    {
        return Err("discovery selection or exact transport request drifted".into());
    }
    discovered_resolver
        .load_module("app/main@2.mlg", std::path::Path::new("/virtual/main.mlg"))
        .map_err(|error| error.to_string())?;

    let mut invalid_discovery = FixtureDiscovery {
        versions: vec!["2".into(), "02".into()],
        requests: Vec::new(),
        failure: None,
    };
    let mut invalid_transport = FixtureTransport {
        envelope: envelope.clone(),
        requests: Vec::new(),
    };
    if discovered_resolver
        .register_trusted_package_from_discovery(
            "app",
            &RegistryPackageVersionPolicy::Exact("2".into()),
            &mut invalid_discovery,
            &mut invalid_transport,
            &policy,
        )
        .is_ok()
        || !invalid_transport.requests.is_empty()
    {
        return Err("invalid discovery candidates reached transport".into());
    }

    let mut untrusted_signature = signature.clone();
    untrusted_signature.key_id = "unknown-key".into();
    if (RegistryPackageIndexEnvelope {
        index: index.clone(),
        signature: untrusted_signature,
    })
    .verify_with_policy(&policy)
    .is_ok()
    {
        return Err("package-index trust policy accepted an unknown key ID".into());
    }

    let mut tampered_signature = signature.clone();
    tampered_signature.key_id = "wrong-key".into();
    if index.verify_signature(&tampered_signature).is_ok() {
        return Err("package-index signature accepted a different key ID".into());
    }

    let mut traversal = package("app", "1", "app/main.mlg");
    traversal.modules[0].path = "app/../escape.mlg".into();
    if (RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![traversal],
    })
    .validate_and_canonicalize()
    .is_ok()
    {
        return Err("path traversal package module was accepted".into());
    }

    let mut cycle_a = package("cyclea", "1", "cyclea/main.mlg");
    let mut cycle_b = package("cycleb", "1", "cycleb/main.mlg");
    cycle_a.dependencies.push(RegistryPackageDependency {
        name: "cycleb".into(),
        version: "1".into(),
        manifest_hash: None,
    });
    cycle_b.dependencies.push(RegistryPackageDependency {
        name: "cyclea".into(),
        version: "1".into(),
        manifest_hash: None,
    });
    cycle_a.manifest_hash = cycle_a
        .canonical_hash()
        .map_err(|error| error.to_string())?
        .to_string();
    cycle_b.manifest_hash = cycle_b
        .canonical_hash()
        .map_err(|error| error.to_string())?
        .to_string();
    if (RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![cycle_a, cycle_b],
    })
    .validate_and_canonicalize()
    .map_err(|error| error.to_string())
    .is_ok()
    {
        return Err("dependency cycle was accepted".into());
    }

    let mut incremental = IncrementalCompiler::new(std::path::Path::new("/virtual"));
    let source = "geometry Main { let solid = Box(1.0, 1.0, 1.0) }";
    let geo = incremental
        .compile_to_geo(source, "main.mlg")
        .map_err(|error| error.to_string())?;
    let python = incremental
        .compile_to_python(source, "main.mlg")
        .map_err(|error| error.to_string())?;
    let incremental_stats = incremental.stats();
    if geo.canonical_ir_hash != python.canonical_ir_hash
        || incremental_stats.artifact_section_entries != 1
        || incremental_stats.artifact_section_hits != 1
        || incremental_stats.artifact_section_misses != 1
    {
        return Err("incremental backend-neutral artifact section cache drifted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("exact_version_checks", 2.0)
        .metric("dependency_order_checks", 3.0)
        .metric("canonical_order_checks", 3.0)
        .metric("manifest_hash_checks", 2.0)
        .metric("path_policy_checks", 1.0)
        .metric("dependency_cycle_checks", 1.0)
        .metric("signature_checks", 5.0)
        .metric("transport_fetch_checks", 3.0)
        .metric("discovery_selection_checks", 5.0)
        .metric("incremental_artifact_section_checks", 3.0)
        .metric("migration_checks", 6.0)
        .metric("checks_passed", 34.0)
        .minimum("exact_version_checks", 2.0)
        .minimum("dependency_order_checks", 3.0)
        .minimum("canonical_order_checks", 3.0)
        .minimum("manifest_hash_checks", 2.0)
        .minimum("path_policy_checks", 1.0)
        .minimum("dependency_cycle_checks", 1.0)
        .minimum("signature_checks", 5.0)
        .minimum("transport_fetch_checks", 3.0)
        .minimum("discovery_selection_checks", 5.0)
        .minimum("incremental_artifact_section_checks", 3.0)
        .minimum("migration_checks", 6.0)
        .input_checksum("source:meshlang-registry-package:v1")
        .reference("exact local package index and dependency policy")
        .reference_checksum("analytic/deterministic-fixture:meshlang-registry-package-v1")
        .dimension(
            "package names, exact versions, module paths, hashes, dependency graph, signatures",
        )
        .formulation("MeshLang deterministic and authenticated registry package index")
        .tolerance(
            "all exact policy and signature checks pass without network or fallback resolution",
        )
        .seed("meshlang-registry-package-reference:cpu:v1"))
}

fn main() {
    let started = std::time::Instant::now();
    let (result, passed) = match run() {
        Ok(result) => (
            result.elapsed_ms(started.elapsed().as_millis() as u64),
            true,
        ),
        Err(error) => (
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
            false,
        ),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
