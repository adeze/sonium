use meshlang_compiler::{CompileResult, compile_to_geo_full, compile_to_python_full};
use serde_json::Value;

const SOURCE: &str = r#"
geometry model {
    // The topology parent must remain traceable through both transforms.
    let parent = Box(1.0, 1.0, 1.0)
    /// First transformed binding.
    let shifted = parent |> translate(1.0, 2.0, 3.0)
    /// Second transformed binding.
    let spun = shifted |> rotate(axis: Z, angle: 0.25)
    /// Independent sibling used by the multi-binding result.
    let cap = Cylinder(r: 0.2, h: 0.5)
    let pair = spun | cap
}

physics {
    // Both topology queries are deliberately rooted in derived bindings.
    group shifted_wall: neumann = model.shifted.faces
    group pair_wall: neumann = model.pair.faces
}
"#;

fn metadata(result: &CompileResult) -> Value {
    serde_json::from_str(&result.backend_metadata_json).expect("backend metadata JSON")
}

fn provenance_for(result: &CompileResult, suffix: &str) -> Vec<Value> {
    metadata(result)["source_map"]["entities"]
        .as_array()
        .expect("source-map entities")
        .iter()
        .find(|entity| {
            entity["name"]
                .as_str()
                .is_some_and(|name| name.ends_with(suffix))
        })
        .and_then(|entity| entity["provenance"].as_array())
        .cloned()
        .expect("transformed entity provenance")
}

fn contains_source_span(provenance: &[Value], source: &str, needle: &str) -> bool {
    let start = source.find(needle).expect("source binding");
    provenance.iter().any(|reference| {
        reference["span"]["start"]
            .as_u64()
            .is_some_and(|offset| offset as usize >= start)
    })
}

fn comments(result: &CompileResult) -> Vec<Value> {
    metadata(result)["source_map"]["source_file"]["comments"]
        .as_array()
        .expect("source-map comments")
        .clone()
}

fn parent_chain_for(result: &CompileResult, suffix: &str) -> Vec<Value> {
    metadata(result)["source_map"]["entities"]
        .as_array()
        .expect("source-map entities")
        .iter()
        .find(|entity| {
            entity["name"]
                .as_str()
                .is_some_and(|name| name.ends_with(suffix))
        })
        .and_then(|entity| entity["parent_chain"].as_array())
        .cloned()
        .expect("transformed entity parent chain")
}

pub fn transformed_parent_topology_has_backend_and_provenance_parity() {
    let geo = compile_to_geo_full(SOURCE, "transformed-parent-topology-packet.mlg")
        .expect("GEO compilation");
    let python = compile_to_python_full(SOURCE, "transformed-parent-topology-packet.mlg")
        .expect("Python compilation");
    let geo_repeat = compile_to_geo_full(SOURCE, "transformed-parent-topology-packet.mlg")
        .expect("repeat GEO compilation");

    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(
        geo.provenance_manifest_json,
        python.provenance_manifest_json
    );
    assert_eq!(
        geo.sensitivity_manifest_json,
        python.sensitivity_manifest_json
    );
    assert_eq!(geo.canonical_ir_json, geo_repeat.canonical_ir_json);
    assert_eq!(geo.backend_metadata_json, geo_repeat.backend_metadata_json);
    let shifted_chain = parent_chain_for(&geo, "model.shifted");
    let spun_chain = parent_chain_for(&geo, "model.spun");
    assert_eq!(shifted_chain.len(), 1, "one direct transformed parent");
    assert_eq!(
        spun_chain.len(),
        2,
        "nested transforms retain full parent chain"
    );
    assert_eq!(
        parent_chain_for(&geo, "model.shifted"),
        parent_chain_for(&python, "model.shifted")
    );
    assert_eq!(
        parent_chain_for(&geo, "model.spun"),
        parent_chain_for(&python, "model.spun")
    );

    // The derived topology paths must reach both text backends rather than
    // falling back to ordinary dictionary/field access.
    assert!(geo.code.contains("shifted_wall"), "{0}", geo.code);
    assert!(geo.code.contains("pair_wall"), "{0}", geo.code);
    assert!(python.code.contains("shifted_wall"), "{0}", python.code);
    assert!(python.code.contains("pair_wall"), "{0}", python.code);
    assert!(
        python.code.contains("_ml_boundary(_ml_dimtags(3,"),
        "{0}",
        python.code
    );
    assert!(geo.canonical_ir_json.contains("model.shifted"));
    assert!(geo.canonical_ir_json.contains("model.pair"));

    let geo_spun = provenance_for(&geo, ".spun");
    let python_spun = provenance_for(&python, ".spun");
    let geo_pair = provenance_for(&geo, ".pair");
    assert_eq!(geo_spun, python_spun);
    assert!(!geo_pair.is_empty(), "multi-binding result lost provenance");
    assert!(contains_source_span(&geo_spun, SOURCE, "let parent"));
    assert!(contains_source_span(&geo_spun, SOURCE, "let shifted"));
    assert!(contains_source_span(&geo_spun, SOURCE, "let spun"));

    let geo_comments = comments(&geo);
    let python_comments = comments(&python);
    assert_eq!(geo_comments, python_comments);
    for text in [
        "// The topology parent must remain traceable through both transforms.",
        "/// First transformed binding.",
        "/// Second transformed binding.",
        "/// Independent sibling used by the multi-binding result.",
    ] {
        assert!(
            geo_comments
                .iter()
                .any(|comment| comment["text"].as_str() == Some(text)),
            "missing source comment: {text}"
        );
    }
}

#[test]
fn transformed_parent_topology_has_backend_and_provenance_parity_test() {
    transformed_parent_topology_has_backend_and_provenance_parity();
}
