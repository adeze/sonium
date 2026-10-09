use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use meshlang_compiler::{
    preview_canonical_ir, preview_canonical_ir_json, CompileOptions, IncrementalCompiler,
    RegistryPackageDiscovery, RegistryPackageDiscoveryError, RegistryPackageIndex,
    RegistryPackageIndexEnvelope, RegistryPackageIndexSignature, RegistryPackageManifest,
    RegistryPackageModule, RegistryPackageTransport, RegistryPackageTransportError,
    RegistryPackageVersionPolicy, RegistrySignaturePolicy, RegistryTrustAnchor,
    REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
};
use std::fs;

#[derive(Debug, Default)]
struct IncrementalHessianProvider {
    requests: std::sync::atomic::AtomicUsize,
}

impl meshlang_compiler::HessianMetricProvider for IncrementalHessianProvider {
    fn resolve(
        &self,
        request: meshlang_compiler::HessianMetricRequest<'_>,
    ) -> Result<[meshlang_ast::Expr; 6], String> {
        self.requests
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(std::array::from_fn(|_| request.field.clone()))
    }
}

#[test]
fn single_source_preview_exposes_versioned_canonical_ir_json() {
    let source = "geometry Main { let solid = Box(1.0, 2.0, 3.0) }";
    let preview = preview_canonical_ir(source, "preview.mlg").unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&preview_canonical_ir_json(source, "preview.mlg").unwrap()).unwrap();

    assert_eq!(json["preview_schema"], 1);
    assert_eq!(
        json["canonical_ir_hash"],
        preview.canonical_ir_hash.to_string()
    );
    assert_eq!(json["canonical_ir"]["document_version"], 1);
    assert_eq!(
        json["canonical_ir"]["ir"]["entities"][0]["name"],
        "Main.solid"
    );
    assert_eq!(preview.canonical_ir.entities.len(), 1);
}

#[test]
fn single_source_preview_rejects_filesystem_imports_explicitly() {
    let error = preview_canonical_ir("import \"part.mlg\"\n", "preview.mlg")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not resolve MeshLang imports"),
        "{error}"
    );
    assert!(error.contains("IncrementalCompiler"), "{error}");
}

#[test]
fn incremental_facade_reuses_and_invalidates_module_and_ir_caches() {
    let directory = tempfile::tempdir().unwrap();
    let module = directory.path().join("part.mlg");
    let root = directory.path().join("root.mlg");
    fs::write(
        &module,
        "pub geometry Part { let solid = Box(1.0, 1.0, 1.0) }\n",
    )
    .unwrap();
    let source = "import \"part.mlg\"\n";
    let options = CompileOptions {
        import_sandbox_root: Some(directory.path().to_path_buf()),
        ..CompileOptions::default()
    };
    let mut compiler = IncrementalCompiler::with_options(directory.path(), options).unwrap();

    let first = compiler
        .preview_canonical_ir(source, root.to_str().unwrap())
        .unwrap();
    assert_eq!(
        compiler.stats(),
        meshlang_compiler::IncrementalCompilerStats {
            parsed_module_entries: 1,
            parsed_module_hits: 0,
            parsed_module_misses: 1,
            canonical_ir_entries: 1,
            canonical_ir_hits: 0,
            canonical_ir_misses: 1,
            backend_artifact_entries: 0,
            backend_artifact_hits: 0,
            backend_artifact_misses: 0,
            artifact_section_entries: 0,
            artifact_section_hits: 0,
            artifact_section_misses: 0,
        }
    );

    let repeated = compiler
        .preview_canonical_ir(source, root.to_str().unwrap())
        .unwrap();
    assert_eq!(first.canonical_ir_hash, repeated.canonical_ir_hash);
    let warm = compiler.stats();
    assert_eq!((warm.parsed_module_hits, warm.canonical_ir_hits), (1, 1));

    fs::write(
        &module,
        "pub geometry Part { let solid = Box(2.0, 1.0, 1.0) }\n",
    )
    .unwrap();
    let changed = compiler
        .preview_canonical_ir(source, root.to_str().unwrap())
        .unwrap();
    assert_ne!(first.canonical_ir_hash, changed.canonical_ir_hash);
    let invalidated = compiler.stats();
    assert_eq!(
        (
            invalidated.parsed_module_misses,
            invalidated.canonical_ir_misses
        ),
        (2, 2)
    );
    assert_eq!(invalidated.canonical_ir_entries, 2);

    assert!(compiler.invalidate_module(&module));
    assert_eq!(compiler.stats().parsed_module_entries, 0);
    compiler.clear_caches();
    assert_eq!(compiler.stats().canonical_ir_entries, 0);
    assert_eq!(compiler.stats().artifact_section_entries, 0);
    compiler.reset_stats();
    assert_eq!(compiler.stats().parsed_module_misses, 0);
}

#[test]
fn incremental_backends_share_the_canonical_plan_cache() {
    let directory = tempfile::tempdir().unwrap();
    let filename = directory.path().join("main.mlg");
    let source = "geometry Main { let solid = Box(1.0, 1.0, 1.0) }";
    let mut compiler = IncrementalCompiler::new(directory.path());

    let geo = compiler
        .compile_to_geo(source, filename.to_str().unwrap())
        .unwrap();
    let python = compiler
        .compile_to_python(source, filename.to_str().unwrap())
        .unwrap();

    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert!(geo.code.contains("Box("));
    assert!(python.code.contains("addBox"));
    let first_stats = compiler.stats();
    assert_eq!(first_stats.canonical_ir_hits, 1);
    assert_eq!(
        (
            first_stats.backend_artifact_entries,
            first_stats.backend_artifact_hits,
            first_stats.backend_artifact_misses,
        ),
        (2, 0, 2)
    );
    assert_eq!(
        (
            first_stats.artifact_section_entries,
            first_stats.artifact_section_hits,
            first_stats.artifact_section_misses,
        ),
        (1, 1, 1)
    );

    let repeated = compiler
        .compile_to_geo(source, filename.to_str().unwrap())
        .unwrap();
    assert_eq!(repeated.code, geo.code);
    let warm_stats = compiler.stats();
    assert_eq!(
        (
            warm_stats.backend_artifact_entries,
            warm_stats.backend_artifact_hits,
            warm_stats.backend_artifact_misses,
        ),
        (2, 1, 2)
    );
}

#[test]
fn incremental_backends_share_hessian_provider_results_and_artifact_cache() {
    let directory = tempfile::tempdir().unwrap();
    let provider = std::sync::Arc::new(IncrementalHessianProvider::default());
    let options = CompileOptions {
        hessian_metric_provider: Some(provider.clone()),
        ..CompileOptions::default()
    };
    let mut compiler = IncrementalCompiler::with_options(directory.path(), options).unwrap();
    let source = r#"
geometry model { let body = Box(1.0, 1.0, 1.0) }
mesh for model {
    metric model.body {
        kind: hessian
        field: 1.0
        minimum_size: 0.01
        maximum_size: 0.1
    }
}
"#;

    let geo = compiler
        .compile_to_geo(source, "incremental-hessian.mlg")
        .unwrap();
    let python = compiler
        .compile_to_python(source, "incremental-hessian.mlg")
        .unwrap();
    let repeated = compiler
        .compile_to_geo(source, "incremental-hessian.mlg")
        .unwrap();

    assert!(geo.code.contains("MathEvalAniso"));
    assert!(python.code.contains("MathEvalAniso"));
    assert_eq!(repeated.code, geo.code);
    assert_eq!(
        provider.requests.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "one provider request per uncached backend artifact"
    );
    let stats = compiler.stats();
    assert_eq!(stats.backend_artifact_entries, 2);
    assert_eq!(stats.backend_artifact_hits, 1);
}

#[test]
fn incremental_facade_registers_package_index_for_imports() {
    let directory = tempfile::tempdir().unwrap();
    let mut package = RegistryPackageManifest {
        name: "pkg".into(),
        version: "1".into(),
        modules: vec![RegistryPackageModule {
            path: "pkg/root@1.mlg".into(),
            source: "pub const ROOT = 1\n".into(),
        }],
        dependencies: Vec::new(),
        manifest_hash: String::new(),
    };
    package.manifest_hash = package.canonical_hash().unwrap().to_string();
    let index = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![package],
    };

    let mut compiler = IncrementalCompiler::new(directory.path());
    compiler.register_package_index(&index).unwrap();
    let preview = compiler
        .preview_canonical_ir(
            "import \"pkg/root@1.mlg\"\n geometry Main { let solid = Box(1.0, 2.0, 3.0) }",
            "main.mlg",
        )
        .unwrap();
    assert!(!preview.canonical_ir.entities.is_empty());
    assert_eq!(compiler.stats().parsed_module_entries, 1);
}

#[derive(Debug)]
struct FixturePackageTransport {
    envelope: Option<RegistryPackageIndexEnvelope>,
    requests: Vec<(String, String)>,
    failure: Option<RegistryPackageTransportError>,
}

impl RegistryPackageTransport for FixturePackageTransport {
    fn fetch_package_index(
        &mut self,
        package: &str,
        version: &str,
    ) -> Result<RegistryPackageIndexEnvelope, RegistryPackageTransportError> {
        self.requests.push((package.to_owned(), version.to_owned()));
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        self.envelope
            .clone()
            .ok_or_else(|| RegistryPackageTransportError::InvalidResponse("empty fixture".into()))
    }
}

#[derive(Debug)]
struct FixturePackageDiscovery {
    versions: Vec<String>,
    requests: Vec<String>,
    failure: Option<RegistryPackageDiscoveryError>,
}

impl RegistryPackageDiscovery for FixturePackageDiscovery {
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

fn transport_fixture() -> (RegistryPackageIndexEnvelope, RegistrySignaturePolicy) {
    let mut package = RegistryPackageManifest {
        name: "remote".into(),
        version: "1".into(),
        modules: vec![RegistryPackageModule {
            path: "remote/root@1.mlg".into(),
            source: "pub geometry Part { let solid = Box(1.0, 1.0, 1.0) }\n".into(),
        }],
        dependencies: Vec::new(),
        manifest_hash: String::new(),
    };
    package.manifest_hash = package.canonical_hash().unwrap().to_string();
    let index = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![package],
    }
    .validate_and_canonicalize()
    .unwrap();

    let signing_key = SigningKey::from([17_u8; 32]);
    let key_id = "fixture-transport-key";
    let signature = signing_key.sign(&index.signing_payload(key_id).unwrap());
    let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
    let hex = |bytes: &[u8]| {
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    let envelope = RegistryPackageIndexEnvelope {
        index,
        signature: RegistryPackageIndexSignature {
            algorithm: "ed25519".into(),
            key_id: key_id.into(),
            public_key: hex(&public_key),
            signature: hex(&signature.to_bytes()),
        },
    };
    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: key_id.into(),
            public_key: hex(&public_key),
        }],
    };
    (envelope, policy)
}

#[test]
fn incremental_facade_fetches_exact_trusted_package_from_transport() {
    let directory = tempfile::tempdir().unwrap();
    let (envelope, policy) = transport_fixture();
    let expected_hash = envelope.index.index_hash().unwrap();
    let mut transport = FixturePackageTransport {
        envelope: Some(envelope),
        requests: Vec::new(),
        failure: None,
    };
    let mut compiler = IncrementalCompiler::new(directory.path());

    let hash = compiler
        .register_trusted_package_from_transport("remote", "1", &mut transport, &policy)
        .unwrap();

    assert_eq!(hash, expected_hash);
    assert_eq!(transport.requests, vec![("remote".into(), "1".into())]);
    let preview = compiler
        .preview_canonical_ir(
            "import \"remote/root@1.mlg\"\n geometry Main { let solid = Box(1.0, 2.0, 3.0) }",
            "main.mlg",
        )
        .unwrap();
    assert!(preview
        .canonical_ir
        .entities
        .iter()
        .any(|entity| { entity.name.contains("Part") || entity.name.contains("Main.solid") }));
}

#[test]
fn discovery_selects_highest_allowed_version_before_trusted_transport() {
    let directory = tempfile::tempdir().unwrap();
    let (envelope, policy) = transport_fixture();
    let expected_hash = envelope.index.index_hash().unwrap();
    let mut discovery = FixturePackageDiscovery {
        versions: vec!["2".into(), "0.9".into(), "1".into(), "1".into()],
        requests: Vec::new(),
        failure: None,
    };
    let mut transport = FixturePackageTransport {
        envelope: Some(envelope),
        requests: Vec::new(),
        failure: None,
    };
    let mut compiler = IncrementalCompiler::new(directory.path());

    let resolution = compiler
        .register_trusted_package_from_discovery(
            "remote",
            &RegistryPackageVersionPolicy::HighestAtMost("1.0".into()),
            &mut discovery,
            &mut transport,
            &policy,
        )
        .unwrap();

    assert_eq!(resolution.package, "remote");
    assert_eq!(resolution.version, "1");
    assert_eq!(resolution.index_hash, expected_hash);
    assert_eq!(discovery.requests, vec!["remote"]);
    assert_eq!(transport.requests, vec![("remote".into(), "1".into())]);
    assert_eq!(compiler.stats().parsed_module_entries, 0);
    compiler
        .preview_canonical_ir(
            "import \"remote/root@1.mlg\"\n geometry Main { let solid = Box(1.0, 2.0, 3.0) }",
            "main.mlg",
        )
        .unwrap();
}

#[test]
fn discovery_rejects_invalid_candidates_and_does_not_call_transport() {
    let directory = tempfile::tempdir().unwrap();
    let (_, policy) = transport_fixture();
    let mut discovery = FixturePackageDiscovery {
        versions: vec!["1".into(), "01".into()],
        requests: Vec::new(),
        failure: None,
    };
    let mut transport = FixturePackageTransport {
        envelope: None,
        requests: Vec::new(),
        failure: None,
    };
    let mut compiler = IncrementalCompiler::new(directory.path());

    let error = compiler
        .register_trusted_package_from_discovery(
            "remote",
            &RegistryPackageVersionPolicy::HighestAtMost("2".into()),
            &mut discovery,
            &mut transport,
            &policy,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid package version"), "{error}");
    assert_eq!(discovery.requests, vec!["remote"]);
    assert!(transport.requests.is_empty());
    assert_eq!(compiler.stats().parsed_module_entries, 0);
}

#[test]
fn discovery_miss_and_host_failure_leave_resolver_unchanged() {
    let (_, policy) = transport_fixture();
    for (failure, expected) in [
        (None, "no discovered version"),
        (
            Some(RegistryPackageDiscoveryError::Unavailable("offline".into())),
            "package discovery unavailable",
        ),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut discovery = FixturePackageDiscovery {
            versions: vec!["2".into()],
            requests: Vec::new(),
            failure,
        };
        let mut transport = FixturePackageTransport {
            envelope: None,
            requests: Vec::new(),
            failure: None,
        };
        let mut compiler = IncrementalCompiler::new(directory.path());
        let error = compiler
            .register_trusted_package_from_discovery(
                "remote",
                &RegistryPackageVersionPolicy::Exact("1".into()),
                &mut discovery,
                &mut transport,
                &policy,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
        assert!(transport.requests.is_empty());
        assert_eq!(compiler.stats().parsed_module_entries, 0);
    }
}

#[test]
fn transport_rejects_mismatched_package_without_mutating_registry() {
    let directory = tempfile::tempdir().unwrap();
    let (mut envelope, policy) = transport_fixture();
    envelope.index.packages[0].name = "other".into();
    envelope.index.packages[0].modules[0].path = "other/root@1.mlg".into();
    envelope.index.packages[0].manifest_hash = envelope.index.packages[0]
        .canonical_hash()
        .unwrap()
        .to_string();
    let mut transport = FixturePackageTransport {
        envelope: Some(envelope),
        requests: Vec::new(),
        failure: None,
    };
    let mut compiler = IncrementalCompiler::new(directory.path());

    let error = compiler
        .register_trusted_package_from_transport("remote", "1", &mut transport, &policy)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not contain exact package"), "{error}");
    assert_eq!(compiler.stats().parsed_module_entries, 0);
}

#[test]
fn transport_errors_are_typed_and_leave_incremental_state_empty() {
    let directory = tempfile::tempdir().unwrap();
    let (_, policy) = transport_fixture();
    let mut transport = FixturePackageTransport {
        envelope: None,
        requests: Vec::new(),
        failure: Some(RegistryPackageTransportError::Unavailable(
            "offline fixture".into(),
        )),
    };
    let mut compiler = IncrementalCompiler::new(directory.path());

    let error = compiler
        .register_trusted_package_from_transport("remote", "1", &mut transport, &policy)
        .unwrap_err()
        .to_string();
    assert!(error.contains("package transport failed"), "{error}");
    assert!(error.contains("offline fixture"), "{error}");
    assert_eq!(compiler.stats().parsed_module_entries, 0);
}

#[test]
fn transport_rejects_invalid_selectors_before_calling_host() {
    let (envelope, policy) = transport_fixture();
    for (package, version) in [
        ("", "1"),
        ("Remote", "1"),
        ("remote", "01"),
        ("remote", "1."),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut compiler = IncrementalCompiler::new(directory.path());
        let mut transport = FixturePackageTransport {
            envelope: Some(envelope.clone()),
            requests: Vec::new(),
            failure: None,
        };

        let error = compiler
            .register_trusted_package_from_transport(package, version, &mut transport, &policy)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid package transport request"),
            "{error}"
        );
        assert!(
            transport.requests.is_empty(),
            "transport was called for {package}@{version}"
        );
        assert_eq!(compiler.stats().parsed_module_entries, 0);
    }
}

#[test]
fn trusted_package_json_preview_is_registry_only_and_authenticated() {
    let (envelope, policy) = transport_fixture();
    let package_json = serde_json::to_string(&envelope).unwrap();
    let policy_json = serde_json::to_string(&policy).unwrap();
    let preview = meshlang_compiler::preview_canonical_ir_with_trusted_package_json(
        "import \"remote/root@1.mlg\"\n geometry Main { let solid = Box(1.0, 2.0, 3.0) }",
        "main.mlg",
        &package_json,
        &policy_json,
    )
    .unwrap();
    let value: serde_json::Value = serde_json::from_str(&preview).unwrap();
    assert_eq!(value["registry_signature_key_id"], "fixture-transport-key");
    assert!(value["registry_manifest_hash"].as_str().is_some());
    assert!(
        value["canonical_ir"]["ir"]["entities"]
            .as_array()
            .unwrap()
            .len()
            >= 2
    );
}
