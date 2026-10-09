//! CPU-only authenticated MeshLang trust-store key-rotation reference.

use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use meshlang_compiler::{
    FilesystemRegistryKeyTransport, HttpRegistryKeyTransport, REGISTRY_KEY_ROTATION_SCHEMA_VERSION,
    RegistryKeyRotation, RegistryKeyRotationEnvelope, RegistryKeyTransport,
    RegistryKeyTransportError, RegistryManifestSignature, RegistrySignaturePolicy,
    RegistryTrustAnchor,
};
use sonium_qa::QaResult;
use std::time::{Duration, Instant};

const CASE_ID: &str = "meshlang.registry-key-transport-reference.v1";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn signed_rotation(issuer: &SigningKey) -> (RegistrySignaturePolicy, RegistryKeyRotationEnvelope) {
    let public_key: [u8; 32] = VerificationKeyBytes::from(issuer).into();
    let public_key = hex(&public_key);
    let mut envelope = RegistryKeyRotationEnvelope {
        schema_version: REGISTRY_KEY_ROTATION_SCHEMA_VERSION,
        rotation: RegistryKeyRotation {
            additions: vec![RegistryTrustAnchor {
                key_id: "next-issuer".into(),
                public_key: "22".repeat(32),
            }],
            revocations: vec!["initial-issuer".into()],
        },
        signature: RegistryManifestSignature {
            algorithm: "ed25519".into(),
            key_id: "initial-issuer".into(),
            public_key: public_key.clone(),
            signature: String::new(),
        },
    };
    let payload = envelope.signing_payload().expect("rotation payload");
    envelope.signature.signature = hex(&issuer.sign(&payload).to_bytes());
    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: "initial-issuer".into(),
            public_key,
        }],
    };
    (policy, envelope)
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let issuer = SigningKey::from([41_u8; 32]);
    let (policy, envelope) = signed_rotation(&issuer);
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let rotation_path = directory.path().join("release-1.json");
    std::fs::write(
        &rotation_path,
        envelope.to_json().map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    let mut transport = FilesystemRegistryKeyTransport::new(directory.path());
    let rotated = policy
        .rotate_from_transport("release-1", &mut transport)
        .map_err(|error| error.to_string())?;
    if rotated.trusted_keys.len() != 1 || rotated.trusted_keys[0].key_id != "next-issuer" {
        return Err("authenticated rotation did not replace the issuer atomically".into());
    }

    let mut old_key_transport = OneRotationTransport {
        envelope: envelope.clone(),
    };
    if !matches!(
        rotated.rotate_from_transport("release-1", &mut old_key_transport),
        Err(RegistryKeyTransportError::Signature(_))
    ) {
        return Err("revoked issuer remained trusted after rotation".into());
    }

    if !matches!(
        transport.fetch_key_rotation("../escape"),
        Err(RegistryKeyTransportError::InvalidRequest(_))
    ) {
        return Err("filesystem selector traversal was accepted".into());
    }
    std::fs::write(directory.path().join("malformed.json"), "{}")
        .map_err(|error| error.to_string())?;
    if !matches!(
        transport.fetch_key_rotation("malformed"),
        Err(RegistryKeyTransportError::InvalidResponse(_))
    ) {
        return Err("malformed rotation response was accepted".into());
    }
    std::fs::write(
        directory.path().join("oversized.json"),
        "x".repeat(64 * 1024 + 1),
    )
    .map_err(|error| error.to_string())?;
    if !matches!(
        transport.fetch_key_rotation("oversized"),
        Err(RegistryKeyTransportError::InvalidResponse(_))
    ) {
        return Err("oversized rotation response was accepted".into());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if !matches!(
            HttpRegistryKeyTransport::new("file:///not-http"),
            Err(RegistryKeyTransportError::InvalidRequest(_))
        ) {
            return Err("HTTP transport accepted a non-HTTP endpoint".into());
        }
        if !matches!(
            HttpRegistryKeyTransport::with_timeout("https://keys.example.test", Duration::ZERO,),
            Err(RegistryKeyTransportError::InvalidRequest(_))
        ) {
            return Err("HTTP transport accepted a zero timeout".into());
        }
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("authenticated_rotation_checks", 1.0)
        .metric("revoked_issuer_rejection_checks", 1.0)
        .metric("selector_rejection_checks", 1.0)
        .metric("malformed_response_rejection_checks", 1.0)
        .metric("oversized_response_rejection_checks", 1.0)
        .metric("http_endpoint_validation_checks", 2.0)
        .dimension("native filesystem authenticated key rotation")
        .input_checksum("temp-store:initial-to-next-issuer:meshlang-key-transport-v1")
        .reference_checksum("ed25519-issuer-rotation-and-bounded-transport:v1")
        .formulation("exact rotation selector, issuer signature, atomic trust-store replacement")
        .tolerance("all signature, revocation, selector, size, and endpoint checks pass")
        .seed("meshlang-registry-key-transport-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
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
