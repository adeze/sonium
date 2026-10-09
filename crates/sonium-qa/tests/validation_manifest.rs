use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[test]
fn validation_manifest_is_structurally_valid_and_points_at_registered_bins() {
    let root = manifest_dir();
    let manifest: toml::Value = fs::read_to_string(root.join("validation-manifest.toml"))
        .expect("read validation-manifest.toml")
        .parse()
        .expect("parse validation-manifest.toml");
    let cases = manifest["case"]
        .as_array()
        .expect("manifest must contain [[case]] entries");
    assert!(!cases.is_empty(), "manifest must not be empty");

    let cargo: toml::Value = fs::read_to_string(root.join("Cargo.toml"))
        .expect("read Cargo.toml")
        .parse()
        .expect("parse Cargo.toml");
    let registered = cargo["bin"]
        .as_array()
        .expect("Cargo.toml must contain [[bin]] entries")
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect::<BTreeSet<_>>();

    let mut ids = BTreeSet::new();
    let allowed_tiers = [
        "smoke",
        "physics-gate",
        "reference-only",
        "deep",
        "expensive",
        "solver-gate",
        "contract-gate",
        "accelerator-gate",
    ];
    for case in cases {
        let table = case.as_table().expect("manifest case must be a table");
        let id = table
            .get("id")
            .and_then(toml::Value::as_str)
            .expect("manifest case id");
        assert!(ids.insert(id), "duplicate validation case id {id}");
        for field in [
            "owner",
            "crate",
            "bin",
            "entry_point",
            "method",
            "scalar",
            "dimension",
            "reference",
            "reference_checksum",
            "norm",
            "tolerance",
            "required_hardware",
            "artifacts",
        ] {
            let value = table
                .get(field)
                .and_then(toml::Value::as_str)
                .unwrap_or_default();
            assert!(!value.trim().is_empty(), "{id} has empty {field}");
        }
        let bin = table["bin"].as_str().expect("manifest case binary");
        assert!(
            registered.contains(bin),
            "{id} references unregistered {bin}"
        );
        let tier = table["tier"].as_str().expect("manifest case tier");
        assert!(
            allowed_tiers.contains(&tier),
            "{id} has unknown tier {tier}"
        );
        let timeout = table["timeout_seconds"]
            .as_integer()
            .expect("manifest case timeout_seconds");
        assert!(timeout > 0, "{id} must have a positive timeout");

        if let Some(features) = table.get("features") {
            let features = features.as_array().expect("features must be an array");
            assert!(
                !features.is_empty()
                    && features.iter().all(|feature| feature
                        .as_str()
                        .is_some_and(|name| !name.trim().is_empty())),
                "{id} has an invalid features list"
            );
        }

        if let Some(paths) = table.get("artifact_paths") {
            let paths = paths.as_array().expect("artifact_paths must be an array");
            let checksums = table
                .get("artifact_checksums")
                .and_then(toml::Value::as_array)
                .expect("artifact_paths requires artifact_checksums");
            assert_eq!(
                paths.len(),
                checksums.len(),
                "{id} artifact paths/checksums must have equal lengths"
            );
            assert!(!paths.is_empty(), "{id} must declare at least one artifact");
            assert_eq!(
                table
                    .get("artifact_checksum_algorithm")
                    .and_then(toml::Value::as_str),
                Some("fnv1a64"),
                "{id} must declare the supported checksum algorithm"
            );
            for (path, checksum) in paths.iter().zip(checksums) {
                let path = path.as_str().expect("artifact path must be a string");
                let checksum = checksum
                    .as_str()
                    .expect("artifact checksum must be a string");
                let bytes = fs::read(root.join(path))
                    .unwrap_or_else(|error| panic!("{id} artifact {path} is unreadable: {error}"));
                assert_eq!(
                    checksum,
                    format!("fnv1a64:{:016x}", fnv1a64(&bytes)),
                    "{id} artifact {path} checksum mismatch"
                );
            }
        }
    }
}

#[test]
fn native_speaker_golden_manifest_matches_schema_v3_contract() {
    let root = manifest_dir();
    let manifest: toml::Value = fs::read_to_string(root.join("validation-manifest.toml"))
        .expect("read validation-manifest.toml")
        .parse()
        .expect("parse validation-manifest.toml");
    let case = manifest["case"]
        .as_array()
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case["id"].as_str() == Some("speaker.native-x86-golden.v1"))
        })
        .expect("manifest must register the native speaker golden case");

    assert_eq!(
        case["bin"].as_str(),
        Some("qa-speaker-native-x86-golden"),
        "native speaker golden case must use its producer binary"
    );
    for field in ["reference", "artifacts"] {
        let value = case[field]
            .as_str()
            .unwrap_or_else(|| panic!("native speaker case must define {field}"));
        assert!(
            value.contains("schema-v3"),
            "native speaker case {field} must describe the producer's schema-v3 artifact"
        );
    }
}
