use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use meshlang_compiler::{
    preview_canonical_ir_with_registry_json, preview_canonical_ir_with_signed_registry_json,
    preview_canonical_ir_with_trusted_registry_json, registry_manifest_signing_payload,
    registry_preview_manifest_hash, verify_registry_manifest_signature, RegistryKeyRotation,
    RegistryKeyRotationEnvelope, RegistryKeyTransport, RegistryKeyTransportError,
    RegistryManifestSignature, RegistryPreviewModule, RegistrySignatureError,
    RegistrySignaturePolicy, RegistryTrustAnchor, REGISTRY_PREVIEW_SCHEMA_VERSION,
};
use serde_json::json;

#[test]
fn registry_preview_flattens_nested_source_without_filesystem_access() {
    let registry = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "modules": [{
            "path": "pkg/part@1.mlg",
            "source": "/// registry part\npub const PART = 1\n"
        }]
    })
    .to_string();
    let output = preview_canonical_ir_with_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
    )
    .expect("registry preview should compile");
    let envelope: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(envelope["preview_schema"], 1);
    assert_eq!(envelope["canonical_ir"]["document_version"], 1);
    assert!(envelope["canonical_ir_hash"].is_string());
    assert!(!envelope["canonical_ir"]["ir"]["entities"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn registry_key_rotation_is_atomic_and_deterministic() {
    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: "old".into(),
            public_key: "11".repeat(32),
        }],
    };
    let rotated = policy
        .rotate(&RegistryKeyRotation {
            additions: vec![RegistryTrustAnchor {
                key_id: "new".into(),
                public_key: "22".repeat(32),
            }],
            revocations: vec!["old".into()],
        })
        .expect("valid rotation");
    assert_eq!(
        rotated
            .trusted_keys
            .iter()
            .map(|anchor| anchor.key_id.as_str())
            .collect::<Vec<_>>(),
        vec!["new"]
    );

    let unknown = policy
        .rotate(&RegistryKeyRotation {
            additions: Vec::new(),
            revocations: vec!["missing".into()],
        })
        .unwrap_err();
    assert!(matches!(unknown, RegistrySignatureError::UnknownKeyId(id) if id == "missing"));

    let empty = policy
        .rotate(&RegistryKeyRotation {
            additions: Vec::new(),
            revocations: vec!["old".into()],
        })
        .unwrap_err();
    assert!(matches!(empty, RegistrySignatureError::EmptyTrustStore));
}

#[test]
fn registry_preview_rejects_unregistered_filesystem_and_unknown_versions() {
    let registry = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "modules": [{
            "path": "pkg/part@1.mlg",
            "source": "pub const PART = 1\n"
        }]
    })
    .to_string();
    let error = preview_canonical_ir_with_registry_json(
        "import \"pkg/part@2.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("filesystem import") || error.contains("register the module"),
        "{error}"
    );
}

#[test]
fn registry_preview_rejects_unknown_schema_and_oversized_payloads() {
    let error = preview_canonical_ir_with_registry_json(
        "geometry model { let body = Box(1.0, 1.0, 1.0) }",
        "preview.mlg",
        r#"{"schema_version":99,"modules":[]}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("schema version"), "{error}");

    let oversized = "x".repeat(8 * 1024 * 1024 + 1);
    let error = preview_canonical_ir_with_registry_json("", "preview.mlg", &oversized)
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds"), "{error}");
}

#[test]
fn registry_manifest_hash_is_order_independent_but_content_sensitive() {
    let first = vec![
        RegistryPreviewModule {
            path: "pkg/a@1.mlg".into(),
            source: "pub const A = 1\n".into(),
        },
        RegistryPreviewModule {
            path: "pkg/b@1.mlg".into(),
            source: "pub const B = 2\n".into(),
        },
    ];
    let mut reversed = first.clone();
    reversed.reverse();
    assert_eq!(
        registry_preview_manifest_hash(&first).unwrap(),
        registry_preview_manifest_hash(&reversed).unwrap()
    );

    reversed[0].source.push_str("// changed\n");
    assert_ne!(
        registry_preview_manifest_hash(&first).unwrap(),
        registry_preview_manifest_hash(&reversed).unwrap()
    );
}

#[test]
fn registry_preview_validates_and_reports_manifest_hash() {
    let modules = vec![RegistryPreviewModule {
        path: "pkg/part@1.mlg".into(),
        source: "pub const PART = 1\n".into(),
    }];
    let manifest_hash = registry_preview_manifest_hash(&modules).unwrap();
    let registry = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "manifest_hash": manifest_hash.to_string(),
        "modules": modules,
    })
    .to_string();
    let output = preview_canonical_ir_with_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
    )
    .unwrap();
    let envelope: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(
        envelope["registry_manifest_hash"],
        manifest_hash.to_string()
    );

    let bad_registry = registry.replace(&manifest_hash.to_string(), &"0".repeat(64));
    let error = preview_canonical_ir_with_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &bad_registry,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("manifest hash does not match"), "{error}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn signed_registry(modules: Vec<RegistryPreviewModule>) -> (String, Vec<RegistryPreviewModule>) {
    let key_id = "qa-key-v1";
    let signing_key = SigningKey::from([7_u8; 32]);
    let payload = registry_manifest_signing_payload(&modules, key_id).unwrap();
    let signature = signing_key.sign(&payload);
    let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
    let request = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "manifest_hash": registry_preview_manifest_hash(&modules).unwrap().to_string(),
        "modules": modules,
        "signature": {
            "algorithm": "ed25519",
            "key_id": key_id,
            "public_key": hex(&public_key),
            "signature": hex(&signature.to_bytes()),
        },
    });
    (request.to_string(), modules)
}

#[test]
fn signed_registry_preview_verifies_before_lowering_and_reports_key() {
    let modules = vec![RegistryPreviewModule {
        path: "pkg/part@1.mlg".into(),
        source: "pub const PART = 1\n".into(),
    }];
    let (registry, modules) = signed_registry(modules);
    let output = preview_canonical_ir_with_signed_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
    )
    .unwrap();
    let envelope: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(envelope["registry_signature_key_id"], "qa-key-v1");

    let value: serde_json::Value = serde_json::from_str(&registry).unwrap();
    let signature: RegistryManifestSignature =
        serde_json::from_value(value["signature"].clone()).unwrap();
    assert_eq!(
        verify_registry_manifest_signature(&modules, &signature).unwrap(),
        registry_preview_manifest_hash(&modules).unwrap()
    );
}

#[test]
fn signed_registry_preview_rejects_missing_and_tampered_signatures() {
    let modules = vec![RegistryPreviewModule {
        path: "pkg/part@1.mlg".into(),
        source: "pub const PART = 1\n".into(),
    }];
    let (registry, _) = signed_registry(modules.clone());
    let unsigned = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "modules": modules,
    })
    .to_string();
    let missing = preview_canonical_ir_with_signed_registry_json(
        "geometry model { Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &unsigned,
    )
    .unwrap_err()
    .to_string();
    assert!(
        missing.contains("requires a manifest signature"),
        "{missing}"
    );

    let tampered = registry.replace("pub const PART = 1", "pub const PART = 2");
    let rejected = preview_canonical_ir_with_signed_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &tampered,
    )
    .unwrap_err()
    .to_string();
    assert!(
        rejected.contains("manifest hash does not match")
            || rejected.contains("signature verification failed"),
        "{rejected}"
    );
}

#[test]
fn registry_signature_validation_rejects_wrong_algorithm_and_hex_shape() {
    let modules = vec![RegistryPreviewModule {
        path: "pkg/part@1.mlg".into(),
        source: "pub const PART = 1\n".into(),
    }];
    let wrong_algorithm = RegistryManifestSignature {
        algorithm: "rsa-sha256".into(),
        key_id: "qa-key-v1".into(),
        public_key: "00".repeat(32),
        signature: "00".repeat(64),
    };
    assert!(matches!(
        verify_registry_manifest_signature(&modules, &wrong_algorithm),
        Err(meshlang_compiler::RegistrySignatureError::UnsupportedAlgorithm)
    ));

    let malformed_hex = RegistryManifestSignature {
        algorithm: "ed25519".into(),
        key_id: "qa-key-v1".into(),
        public_key: "zz".repeat(32),
        signature: "00".repeat(64),
    };
    assert!(matches!(
        verify_registry_manifest_signature(&modules, &malformed_hex),
        Err(meshlang_compiler::RegistrySignatureError::InvalidHex {
            field: "public_key"
        })
    ));
}

#[test]
fn trusted_registry_preview_pins_key_id_and_public_key() {
    let modules = vec![RegistryPreviewModule {
        path: "pkg/part@1.mlg".into(),
        source: "pub const PART = 1\n".into(),
    }];
    let (registry, _) = signed_registry(modules.clone());
    let request: serde_json::Value = serde_json::from_str(&registry).unwrap();
    let public_key = request["signature"]["public_key"].as_str().unwrap();
    let trusted_policy = json!({
        "trusted_keys": [{"key_id": "qa-key-v1", "public_key": public_key}],
    })
    .to_string();
    let output = preview_canonical_ir_with_trusted_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
        &trusted_policy,
    )
    .unwrap();
    let envelope: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(envelope["registry_signature_key_id"], "qa-key-v1");

    let untrusted_policy = json!({
        "trusted_keys": [{"key_id": "other-key", "public_key": public_key}],
    })
    .to_string();
    let error = preview_canonical_ir_with_trusted_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
        &untrusted_policy,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("not trusted"), "{error}");
}
#[test]
fn authenticated_key_rotation_verifies_issuer_and_preserves_policy_on_failure() {
    use meshlang_compiler::{
        RegistryKeyRotationEnvelope, RegistryKeyTransportError,
        REGISTRY_KEY_ROTATION_SCHEMA_VERSION,
    };

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    let issuer = SigningKey::from([7_u8; 32]);
    let issuer_public_key: [u8; 32] = VerificationKeyBytes::from(&issuer).into();
    let new_public_key = "22".repeat(32);
    let rotation = RegistryKeyRotation {
        additions: vec![RegistryTrustAnchor {
            key_id: "new".into(),
            public_key: new_public_key.clone(),
        }],
        revocations: vec!["issuer".into()],
    };
    let mut envelope = RegistryKeyRotationEnvelope {
        schema_version: REGISTRY_KEY_ROTATION_SCHEMA_VERSION,
        rotation,
        signature: RegistryManifestSignature {
            algorithm: "ed25519".into(),
            key_id: "issuer".into(),
            public_key: hex(&issuer_public_key),
            signature: String::new(),
        },
    };
    let payload = envelope.signing_payload().expect("rotation payload");
    envelope.signature.signature = hex(&issuer.sign(&payload).to_bytes());
    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: "issuer".into(),
            public_key: hex(&issuer_public_key),
        }],
    };

    let mut transport = OneRotationTransport {
        envelope: envelope.clone(),
    };
    let rotated = policy
        .rotate_from_transport("release-1", &mut transport)
        .expect("issuer-authenticated rotation");
    assert_eq!(rotated.trusted_keys[0].key_id, "new");
    assert_eq!(rotated.trusted_keys[0].public_key, new_public_key);

    let mut tampered = envelope.clone();
    tampered.rotation.additions[0].public_key = "33".repeat(32);
    let mut tampered_transport = OneRotationTransport { envelope: tampered };
    let error = policy
        .rotate_from_transport("release-1", &mut tampered_transport)
        .expect_err("tampered rotation must fail");
    assert!(matches!(error, RegistryKeyTransportError::Signature(_)));
    assert_eq!(policy.trusted_keys[0].key_id, "issuer");

    let mut untrusted = envelope;
    untrusted.signature.key_id = "attacker".into();
    let mut untrusted_transport = OneRotationTransport {
        envelope: untrusted,
    };
    let error = policy
        .rotate_from_transport("release-1", &mut untrusted_transport)
        .expect_err("untrusted issuer must fail");
    assert!(matches!(error, RegistryKeyTransportError::Signature(_)));
}

#[test]
fn key_rotation_payload_is_deterministic_and_rejects_schema_algorithm_and_key_mismatch() {
    let issuer = SigningKey::from([9_u8; 32]);
    let issuer_public_key: [u8; 32] = VerificationKeyBytes::from(&issuer).into();
    let mut envelope = RegistryKeyRotationEnvelope {
        schema_version: meshlang_compiler::REGISTRY_KEY_ROTATION_SCHEMA_VERSION,
        rotation: RegistryKeyRotation {
            additions: vec![RegistryTrustAnchor {
                key_id: "next".into(),
                public_key: "44".repeat(32),
            }],
            revocations: vec!["issuer".into()],
        },
        signature: RegistryManifestSignature {
            algorithm: "ed25519".into(),
            key_id: "issuer".into(),
            public_key: issuer_public_key
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            signature: String::new(),
        },
    };
    let first_payload = envelope.signing_payload().expect("rotation payload");
    let second_payload = envelope.signing_payload().expect("rotation payload repeat");
    assert_eq!(first_payload, second_payload);
    let mut reordered = envelope.clone();
    reordered.rotation.additions.push(RegistryTrustAnchor {
        key_id: "another".into(),
        public_key: "55".repeat(32),
    });
    reordered.rotation.revocations.push("backup".into());
    let canonical_payload = reordered.signing_payload().expect("canonical payload");
    reordered.rotation.additions.reverse();
    reordered.rotation.revocations.reverse();
    assert_eq!(
        canonical_payload,
        reordered
            .signing_payload()
            .expect("reordered canonical payload")
    );
    let serialized = envelope.to_json().expect("rotation JSON");
    assert_eq!(
        RegistryKeyRotationEnvelope::from_json(&serialized).expect("round trip"),
        envelope
    );

    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: "issuer".into(),
            public_key: envelope.signature.public_key.clone(),
        }],
    };
    envelope.signature.signature = issuer
        .sign(&first_payload)
        .to_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let mut wrong_schema = envelope.clone();
    wrong_schema.schema_version = 99;
    assert!(matches!(
        policy.verify_key_rotation(&wrong_schema),
        Err(RegistrySignatureError::UnsupportedSchemaVersion(99))
    ));

    let mut wrong_algorithm = envelope.clone();
    wrong_algorithm.signature.algorithm = "rsa-sha256".into();
    assert!(matches!(
        policy.verify_key_rotation(&wrong_algorithm),
        Err(RegistrySignatureError::UnsupportedAlgorithm)
    ));

    let mut wrong_key = envelope;
    wrong_key.signature.public_key = "00".repeat(32);
    assert!(matches!(
        policy.verify_key_rotation(&wrong_key),
        Err(RegistrySignatureError::PublicKeyMismatch { .. })
    ));

    let oversized = "x".repeat(meshlang_compiler::MAX_REGISTRY_KEY_ROTATION_BYTES + 1);
    assert!(matches!(
        RegistryKeyRotationEnvelope::from_json(&oversized),
        Err(RegistrySignatureError::PayloadTooLarge { .. })
    ));
    let mut oversized_envelope = RegistryKeyRotationEnvelope {
        rotation: RegistryKeyRotation {
            additions: Vec::new(),
            revocations: vec!["x".repeat(meshlang_compiler::MAX_REGISTRY_KEY_ROTATION_BYTES)],
        },
        ..RegistryKeyRotationEnvelope::from_json(&serialized).expect("rotation round trip")
    };
    oversized_envelope.signature.signature.clear();
    assert!(matches!(
        oversized_envelope.to_json(),
        Err(RegistrySignatureError::PayloadTooLarge { .. })
    ));
}

struct OneRotationTransport {
    envelope: RegistryKeyRotationEnvelope,
}

impl RegistryKeyTransport for OneRotationTransport {
    fn fetch_key_rotation(
        &mut self,
        _rotation_id: &str,
    ) -> Result<RegistryKeyRotationEnvelope, RegistryKeyTransportError> {
        Ok(self.envelope.clone())
    }
}

#[test]
fn filesystem_key_transport_is_bounded_and_rejects_selectors_and_malformed_json() {
    use meshlang_compiler::{
        FilesystemRegistryKeyTransport, RegistryKeyTransport, RegistryKeyTransportError,
    };
    use std::fs;

    let directory = tempfile::tempdir().expect("temporary key store");
    let mut transport = FilesystemRegistryKeyTransport::new(directory.path());
    assert!(matches!(
        transport.fetch_key_rotation("../escape"),
        Err(RegistryKeyTransportError::InvalidRequest(_))
    ));
    assert!(matches!(
        transport.fetch_key_rotation("nested/release"),
        Err(RegistryKeyTransportError::InvalidRequest(_))
    ));

    fs::write(directory.path().join("bad.json"), "{}").expect("malformed fixture");
    assert!(matches!(
        transport.fetch_key_rotation("bad"),
        Err(RegistryKeyTransportError::InvalidResponse(_))
    ));

    let oversized = "x".repeat(64 * 1024 + 1);
    fs::write(directory.path().join("oversized.json"), oversized).expect("oversized fixture");
    assert!(matches!(
        transport.fetch_key_rotation("oversized"),
        Err(RegistryKeyTransportError::InvalidResponse(_))
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn http_key_transport_validates_endpoint_and_selector_before_network_io() {
    use meshlang_compiler::{
        HttpRegistryKeyTransport, RegistryKeyTransport, RegistryKeyTransportError,
    };
    use std::time::Duration;

    assert!(matches!(
        HttpRegistryKeyTransport::new("file:///tmp/keys"),
        Err(RegistryKeyTransportError::InvalidRequest(_))
    ));
    assert!(matches!(
        HttpRegistryKeyTransport::with_timeout("https://keys.example.test", Duration::ZERO),
        Err(RegistryKeyTransportError::InvalidRequest(_))
    ));

    let mut transport =
        HttpRegistryKeyTransport::new("https://keys.example.test").expect("valid HTTP endpoint");
    assert!(matches!(
        transport.fetch_key_rotation("../escape"),
        Err(RegistryKeyTransportError::InvalidRequest(_))
    ));
}
