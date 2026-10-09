//! Release-facing authenticated MeshLang source-registry reference.

use std::time::Instant;

use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use meshlang_compiler::{
    CanonicalIrPreviewEnvelope, REGISTRY_PREVIEW_SCHEMA_VERSION, RegistryKeyRotation,
    RegistryManifestSignature, RegistryPreviewModule, RegistrySignaturePolicy, RegistryTrustAnchor,
    preview_canonical_ir_with_signed_registry_json,
    preview_canonical_ir_with_trusted_registry_json, registry_manifest_signing_payload,
    registry_preview_manifest_hash, verify_registry_manifest_signature,
};
use serde_json::json;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.registry-signature-reference.v1";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let modules = vec![RegistryPreviewModule {
        path: "pkg/part@1.mlg".into(),
        source: "/// signed registry part\npub const PART = 1\n".into(),
    }];
    let key_id = "qa-key-v1";
    let signing_key = SigningKey::from([7_u8; 32]);
    let payload = registry_manifest_signing_payload(&modules, key_id)
        .map_err(|error| format!("signing payload: {error}"))?;
    let detached = signing_key.sign(&payload);
    let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
    let manifest_hash = registry_preview_manifest_hash(&modules)
        .map_err(|error| format!("manifest hash: {error}"))?;
    let registry = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "manifest_hash": manifest_hash.to_string(),
        "modules": modules,
        "signature": {
            "algorithm": "ed25519",
            "key_id": key_id,
            "public_key": hex(&public_key),
            "signature": hex(&detached.to_bytes()),
        },
    })
    .to_string();

    let output = preview_canonical_ir_with_signed_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
    )
    .map_err(|error| format!("signed preview: {error}"))?;
    let envelope = CanonicalIrPreviewEnvelope::from_json(&output)
        .map_err(|error| format!("preview envelope: {error}"))?;
    if envelope.registry_signature_key_id.as_deref() != Some(key_id)
        || envelope.registry_manifest_hash.as_deref() != Some(manifest_hash.to_string().as_str())
    {
        return Err("signed preview envelope lost signature identity".into());
    }
    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: "old-key".into(),
            public_key: "11".repeat(32),
        }],
    };
    let rotated = policy
        .rotate(&RegistryKeyRotation {
            additions: vec![RegistryTrustAnchor {
                key_id: "new-key".into(),
                public_key: "22".repeat(32),
            }],
            revocations: vec!["old-key".into()],
        })
        .map_err(|error| format!("key rotation: {error}"))?;
    if rotated.trusted_keys.len() != 1 || rotated.trusted_keys[0].key_id != "new-key" {
        return Err("key rotation did not atomically replace the old key".into());
    }
    let checks = 4_u64;

    let request: serde_json::Value =
        serde_json::from_str(&registry).map_err(|error| format!("registry JSON: {error}"))?;
    let signature: RegistryManifestSignature = serde_json::from_value(request["signature"].clone())
        .map_err(|error| format!("signature JSON: {error}"))?;
    let verified_hash = verify_registry_manifest_signature(&modules, &signature)
        .map_err(|error| format!("direct verification: {error}"))?;
    if verified_hash != manifest_hash {
        return Err("direct verification returned the wrong manifest hash".into());
    }
    let checks = checks + 1;

    let trusted_policy = json!({
        "trusted_keys": [{
            "key_id": key_id,
            "public_key": request["signature"]["public_key"],
        }],
    })
    .to_string();
    preview_canonical_ir_with_trusted_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
        &trusted_policy,
    )
    .map_err(|error| format!("trusted preview: {error}"))?;
    let checks = checks + 1;

    let untrusted_policy = json!({
        "trusted_keys": [{
            "key_id": "other-key",
            "public_key": request["signature"]["public_key"],
        }],
    })
    .to_string();
    let trust_error = preview_canonical_ir_with_trusted_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &registry,
        &untrusted_policy,
    )
    .expect_err("untrusted key must be rejected")
    .to_string();
    if !trust_error.contains("not trusted") {
        return Err(format!("unexpected trust-policy error: {trust_error}"));
    }
    let checks = checks + 1;

    let missing_signature = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "modules": modules,
    })
    .to_string();
    let missing_error = preview_canonical_ir_with_signed_registry_json(
        "geometry model { Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &missing_signature,
    )
    .expect_err("signed preview must reject missing signatures")
    .to_string();
    if !missing_error.contains("requires a manifest signature") {
        return Err(format!(
            "unexpected missing-signature error: {missing_error}"
        ));
    }
    let checks = checks + 1;

    let tampered = registry.replace("signed registry part", "tampered registry part");
    let tampered_error = preview_canonical_ir_with_signed_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &tampered,
    )
    .expect_err("signed preview must reject tampered modules")
    .to_string();
    if !tampered_error.contains("manifest hash does not match")
        && !tampered_error.contains("signature verification failed")
    {
        return Err(format!("unexpected tamper error: {tampered_error}"));
    }
    let checks = checks + 1;

    Ok(QaResult::passed(CASE_ID)
        .metric("checks", checks as f64)
        .minimum("checks", 7.0)
        .input_checksum("source:meshlang-registry-signature-fixture:v1")
        .reference_checksum("analytic/ed25519-manifest-verification:meshlang-registry-signature-v1")
        .formulation("domain-separated Ed25519 source-registry manifest verification")
        .tolerance("exact signature, manifest, schema, and strict-rejection checks")
        .seed("meshlang-registry-signature-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let started = Instant::now();
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
