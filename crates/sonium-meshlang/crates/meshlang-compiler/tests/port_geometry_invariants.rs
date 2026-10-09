//! Port-geometry invariants across backends and malformed-import admission.
//!
//! Text snapshots alone cannot catch a wrong but stable geometric result, so
//! these tests compare emitted geometry (offsets, region identities) across
//! the Gmsh and Python backends instead of snapshotting one backend's text.
//! Malformed imports must be bounded compile errors, never panics or hangs.

use meshlang_compiler::{compile_to_geo, compile_to_geo_full, compile_to_python_full};

const TRANSFORMED_PORTS: &str = r#"
geometry model {
    let parent = Box(1.0, 1.0, 1.0)
    let shifted = parent |> translate(4.0, 5.0, 6.0)
    let cap = Cylinder(r: 0.2, h: 0.5)
    let pair = shifted | cap
}

physics {
    group shifted_wall: neumann = model.shifted.faces
    group pair_wall: neumann = model.pair.faces
}
"#;

#[test]
fn translated_port_keeps_offsets_and_identity_in_both_backends() {
    let geo = compile_to_geo_full(TRANSFORMED_PORTS, "translated-ports.mlg")
        .expect("translated ports must lower through GEO");
    let python = compile_to_python_full(TRANSFORMED_PORTS, "translated-ports.mlg")
        .expect("translated ports must lower through Python");

    // Same canonical IR from both backends: a passing preview in one backend
    // is not evidence the production mesh assignment is valid unless the
    // other backend agrees on the lowered geometry.
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);

    // Port/region identities survive lowering and the union.
    for identity in ["model.shifted", "model.cap", "model.pair"] {
        assert!(
            geo.canonical_ir_json.contains(identity),
            "canonical IR must name {identity}"
        );
    }

    // The translation offsets reach emitted geometry numerically, in backend
    // syntax, rather than only surviving as IR text.
    assert!(
        geo.code.contains("Translate {4.0, 5.0, 6.0}"),
        "GEO output must carry the translation offsets, got:\n{}",
        geo.code
    );
    assert!(
        python.code.contains("occ.translate"),
        "Python output must translate the solid, got:\n{}",
        python.code
    );
    for offset in ["4.0", "5.0", "6.0"] {
        assert!(
            python.code.contains(offset),
            "Python output must carry offset {offset}, got:\n{}",
            python.code
        );
    }
}

fn write_mlg(dir: &std::path::Path, name: &str, contents: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write fixture");
    path
}

#[test]
fn missing_import_is_a_clean_error_not_a_panic() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let main = write_mlg(
        tmp.path(),
        "main.mlg",
        "import \"does-not-exist.mlg\"\n\ngeometry room {\n    let walls = Box(1.0, 2.0, 3.0)\n}\n",
    );
    let source = std::fs::read_to_string(&main).expect("read main");
    let error = compile_to_geo(&source, main.to_str().expect("utf8 path"))
        .expect_err("missing import must fail compilation");
    assert!(
        !error.to_string().is_empty(),
        "missing import must explain itself"
    );
}

#[test]
fn circular_import_is_a_bounded_error_not_a_hang() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_mlg(
        tmp.path(),
        "a.mlg",
        "import \"b.mlg\"\n\ngeometry a {\n    let body = Box(1.0, 1.0, 1.0)\n}\n",
    );
    let main = write_mlg(
        tmp.path(),
        "b.mlg",
        "import \"a.mlg\"\n\ngeometry b {\n    let body = Box(1.0, 1.0, 1.0)\n}\n",
    );
    let source = std::fs::read_to_string(&main).expect("read main");
    let error = compile_to_geo(&source, main.to_str().expect("utf8 path"))
        .expect_err("circular import must fail compilation");
    assert!(
        !error.to_string().is_empty(),
        "circular import must explain itself"
    );
}
