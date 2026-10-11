use meshlang_compiler::{
    compile_to_geo_full_with_options, compile_to_python_full_with_options, CompileOptions,
};
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

const MODULE_SOURCE: &str = r#"
// Imported topology remains source-addressable after flattening.
pub geometry Imported {
    /// Imported parent binding.
    let parent = Box(1.0, 1.0, 1.0)
    /// Imported first transform.
    let shifted = parent |> translate(1.0, 2.0, 3.0)
    /// Imported nested transform.
    let spun = shifted |> rotate(axis: Z, angle: 0.25)
}
"#;

fn entity_metadata(result: &meshlang_compiler::CompileResult) -> Value {
    serde_json::from_str(&result.backend_metadata_json).expect("backend metadata JSON")
}

fn entity<'a>(metadata: &'a Value, suffix: &str) -> &'a Value {
    metadata["source_map"]["entities"]
        .as_array()
        .expect("source-map entities")
        .iter()
        .find(|entry| {
            entry["name"]
                .as_str()
                .is_some_and(|name| name.ends_with(suffix))
        })
        .unwrap_or_else(|| panic!("missing imported entity {suffix}"))
}

pub fn imported_nested_transforms_preserve_comments_spans_and_parent_chain() {
    let directory = tempdir().expect("temporary import directory");
    let module = directory.path().join("imported.mlg");
    let root = directory.path().join("root.mlg");
    fs::write(&module, MODULE_SOURCE).expect("write imported module");
    let root_source = "import \"imported.mlg\"\n";
    let options = CompileOptions {
        import_sandbox_root: Some(directory.path().to_path_buf()),
        ..CompileOptions::default()
    };

    let geo = compile_to_geo_full_with_options(
        root_source,
        root.to_str().expect("root path UTF-8"),
        &options,
    )
    .expect("GEO imported compile");
    let python = compile_to_python_full_with_options(
        root_source,
        root.to_str().expect("root path UTF-8"),
        &options,
    )
    .expect("Python imported compile");

    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    let canonical_ir =
        serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
            .expect("versioned canonical IR");
    let source_names = canonical_ir
        .ir
        .source_identities
        .iter()
        .map(|identity| identity.display_name.as_str())
        .collect::<Vec<_>>();
    assert!(source_names.iter().any(|name| name.ends_with("root.mlg")));
    assert!(source_names
        .iter()
        .any(|name| name.ends_with("imported.mlg")));
    assert_eq!(
        geo.provenance_manifest_json,
        python.provenance_manifest_json
    );
    assert_eq!(
        geo.sensitivity_manifest_json,
        python.sensitivity_manifest_json
    );

    let geo_metadata = entity_metadata(&geo);
    let shifted = entity(&geo_metadata, "Imported.shifted");
    let spun = entity(&geo_metadata, "Imported.spun");
    assert_eq!(shifted["parent_chain"].as_array().map(Vec::len), Some(1));
    assert_eq!(spun["parent_chain"].as_array().map(Vec::len), Some(2));
    assert!(shifted["provenance"]
        .as_array()
        .is_some_and(|items| !items.is_empty()));
    assert!(spun["provenance"]
        .as_array()
        .is_some_and(|items| !items.is_empty()));

    let comments = geo_metadata["source_map"]["source_file"]["comments"]
        .as_array()
        .expect("flattened imported comments");
    for text in [
        "// Imported topology remains source-addressable after flattening.",
        "/// Imported parent binding.",
        "/// Imported first transform.",
        "/// Imported nested transform.",
    ] {
        assert!(
            comments.iter().any(|comment| comment["text"] == text),
            "missing {text}"
        );
    }
    let imported_file_id = comments
        .iter()
        .find_map(|comment| comment["span"]["file_id"].as_u64())
        .expect("imported comment file identity");
    assert_ne!(
        imported_file_id, 0,
        "imported source should retain a distinct file ID"
    );

    let python_metadata: Value =
        serde_json::from_str(&python.backend_metadata_json).expect("Python backend metadata JSON");
    assert_eq!(
        python_metadata["source_map"]["source_file"]["comments"],
        geo_metadata["source_map"]["source_file"]["comments"]
    );
}

#[test]
fn imported_nested_transforms_preserve_comments_spans_and_parent_chain_test() {
    imported_nested_transforms_preserve_comments_spans_and_parent_chain();
}
