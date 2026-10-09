use meshlang_ast::{CommentKind, ProvenanceOperation, SourceMap};
use meshlang_compiler::{compile_to_geo_full, compile_to_python_full, CompileResult};
use serde_json::Value;
use std::fs;
use std::path::Path;

fn backend_source_map(result: &CompileResult) -> SourceMap {
    let metadata: Value =
        serde_json::from_str(&result.backend_metadata_json).expect("backend metadata JSON");
    serde_json::from_value(metadata["source_map"]["source_file"].clone())
        .expect("typed source map in backend metadata")
}

fn assert_valid_canonical_source_map(source_map: &SourceMap) {
    source_map
        .validate()
        .expect("compiler sidecar source map satisfies provenance contract");
    assert_eq!(
        source_map
            .canonicalized()
            .expect("valid source map can be canonicalized"),
        *source_map,
        "compiler sidecar source map is already canonical"
    );
}

fn provenance_source_map(result: &CompileResult) -> SourceMap {
    let manifest: Value =
        serde_json::from_str(&result.provenance_manifest_json).expect("provenance manifest JSON");
    serde_json::from_value(manifest["source_map"].clone())
        .expect("typed source map in provenance manifest")
}

fn backend_entity_provenance(result: &CompileResult, suffix: &str) -> Vec<Value> {
    let metadata: Value =
        serde_json::from_str(&result.backend_metadata_json).expect("backend metadata JSON");
    metadata["source_map"]["entities"]
        .as_array()
        .expect("backend source-map entities")
        .iter()
        .find(|entity| {
            entity["name"]
                .as_str()
                .is_some_and(|name| name.ends_with(suffix))
        })
        .and_then(|entity| entity["provenance"].as_array())
        .cloned()
        .expect("entity provenance")
}

#[test]
fn backend_sidecars_round_trip_imported_comments_and_provenance() {
    let directory = tempfile::tempdir().expect("temporary MeshLang directory");
    let root_path = directory.path().join("root.mlg");
    let imported_path = directory.path().join("part.mlg");
    let root_source = r#"
        // root import
        import "part.mlg"

        /// root geometry
        geometry model {
            // root body
            let body = Box(2.0, 2.0, 2.0)
        }
    "#;
    let imported_source = r#"
        /// imported geometry
        pub geometry Part {
            // imported body
            let body = Box(1.0, 1.0, 1.0)
        }
    "#;
    fs::write(&root_path, root_source).expect("write root source");
    fs::write(&imported_path, imported_source).expect("write imported source");

    let filename = root_path.to_str().expect("UTF-8 temporary path");
    let geo = compile_to_geo_full(root_source, filename).expect("GEO compilation");
    let python = compile_to_python_full(root_source, filename).expect("Python compilation");

    let geo_map = backend_source_map(&geo);
    let python_map = backend_source_map(&python);
    assert_valid_canonical_source_map(&geo_map);
    assert_valid_canonical_source_map(&python_map);
    assert_eq!(
        geo_map, python_map,
        "both native backends must carry the same source map"
    );
    assert_eq!(geo_map, provenance_source_map(&geo));
    assert_eq!(python_map, provenance_source_map(&python));

    let round_trip = serde_json::from_slice::<SourceMap>(
        &serde_json::to_vec(&geo_map).expect("serialize source map"),
    )
    .expect("deserialize source map");
    assert_eq!(round_trip, geo_map);

    let identities = geo_map
        .source_identities
        .iter()
        .map(|identity| identity.display_name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(identities.len(), 2);
    assert!(identities
        .iter()
        .any(|name| Path::new(name).ends_with("root.mlg")));
    assert!(identities
        .iter()
        .any(|name| Path::new(name).ends_with("part.mlg")));

    let imported_id = geo_map
        .source_identities
        .iter()
        .find(|identity| Path::new(&identity.display_name).ends_with("part.mlg"))
        .expect("imported source identity")
        .id;
    let imported_comment = geo_map
        .comments
        .iter()
        .find(|comment| {
            comment.kind == CommentKind::DocLine
                && comment.text == "/// imported geometry"
                && comment.span.file_id != 0
        })
        .expect("imported doc comment");
    let imported_expansion = geo_map
        .provenance
        .records
        .iter()
        .find(|record| {
            record.operation == ProvenanceOperation::Expanded
                && record
                    .sources
                    .iter()
                    .any(|source| source.source == imported_id)
        })
        .expect("imported expansion provenance");
    let imported_source_node = imported_expansion
        .sources
        .iter()
        .find(|source| source.source == imported_id)
        .expect("imported source provenance")
        .node;
    assert_eq!(imported_comment.target, Some(imported_expansion.node));
    assert_ne!(imported_comment.target, Some(imported_source_node));
    assert!(geo_map.comments.iter().any(|comment| {
        comment.kind == CommentKind::DocLine
            && comment.text == "/// root geometry"
            && comment.span.file_id == 0
            && comment.target.is_some()
    }));
    assert!(geo_map
        .comments
        .iter()
        .any(|comment| { comment.text == "// imported body" && comment.span.file_id != 0 }));
    assert!(geo_map
        .comments
        .iter()
        .any(|comment| { comment.text == "// root body" && comment.span.file_id == 0 }));

    assert!(geo_map.provenance.records.iter().any(|record| {
        record.operation == ProvenanceOperation::Parsed
            && record
                .sources
                .iter()
                .any(|source| source.source == imported_id)
    }));
    assert!(geo_map.provenance.records.iter().any(|record| {
        record.operation == ProvenanceOperation::Expanded
            && record
                .sources
                .iter()
                .any(|source| source.source == imported_id)
    }));
}

#[test]
fn transformed_entities_round_trip_parent_provenance_and_comments() {
    let source = r#"
        geometry model {
            // source parent
            let base = Box(1.0, 1.0, 1.0)
            /// transformed child
            let moved = base
                |> translate(1.0, 2.0, 3.0)
        }
    "#;
    let geo = compile_to_geo_full(source, "transformed-provenance.mlg").expect("GEO compilation");
    let python =
        compile_to_python_full(source, "transformed-provenance.mlg").expect("Python compilation");

    let geo_provenance = backend_entity_provenance(&geo, ".moved");
    let python_provenance = backend_entity_provenance(&python, ".moved");
    assert_eq!(geo_provenance, python_provenance);
    assert!(
        geo_provenance.len() >= 2,
        "transformed entity lost its parent"
    );

    let base_start = source.find("let base").expect("base binding")
        ..source.find("let moved").expect("moved binding");
    let moved_start = source.find("let moved").expect("moved binding");
    assert!(geo_provenance.iter().any(|reference| {
        let span = &reference["span"];
        span["start"]
            .as_u64()
            .is_some_and(|start| base_start.contains(&(start as usize)))
    }));
    assert!(geo_provenance.iter().any(|reference| {
        reference["span"]["start"]
            .as_u64()
            .is_some_and(|start| start as usize >= moved_start)
    }));

    let geo_map = backend_source_map(&geo);
    let python_map = backend_source_map(&python);
    assert_valid_canonical_source_map(&geo_map);
    assert_valid_canonical_source_map(&python_map);
    assert_eq!(geo_map, python_map);
    assert!(geo_map
        .comments
        .iter()
        .any(|comment| comment.text == "// source parent"));
    assert!(geo_map
        .comments
        .iter()
        .any(|comment| comment.text == "/// transformed child"));
    let round_trip = serde_json::from_slice::<SourceMap>(
        &serde_json::to_vec(&geo_map).expect("serialize source map"),
    )
    .expect("deserialize source map");
    assert_eq!(round_trip, geo_map);
}
