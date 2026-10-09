use meshlang_ast::{VersionedAst, CURRENT_AST_SCHEMA_VERSION};
use std::fs;
use std::process::Command;

fn meshlang() -> Command {
    Command::new(env!("CARGO_BIN_EXE_meshlang"))
}

#[test]
fn usage_reports_the_package_version() {
    let output = meshlang().output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains(concat!("MeshLang compiler v", env!("CARGO_PKG_VERSION"))));
}

#[test]
fn parse_emits_the_versioned_ast_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("model.mlg");
    fs::write(&input, "const size = 1.0\n").unwrap();

    let output = meshlang().arg("parse").arg(&input).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let document: VersionedAst = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document.schema_version, CURRENT_AST_SCHEMA_VERSION);
    assert_eq!(document.source_file.declarations.len(), 1);
}

#[test]
fn compile_writes_backend_artifact_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("model.mlg");
    fs::write(
        &input,
        "geometry model {\n  let domain = Box(1.0, 1.0, 1.0)\n}\n",
    )
    .unwrap();

    for (target, extension, expects_map) in [("geo", "geo", true), ("python", "py", false)] {
        let output_path = dir.path().join(format!("model.{extension}"));
        let output = meshlang()
            .arg("compile")
            .arg(&input)
            .arg("--target")
            .arg(target)
            .arg("-o")
            .arg(&output_path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output_path.is_file());

        for suffix in [
            "ir.json",
            "sensitivity.json",
            "provenance.json",
            "metadata.json",
        ] {
            assert!(
                output_path
                    .with_extension(format!("{extension}.{suffix}"))
                    .is_file(),
                "missing {suffix} for {target}"
            );
        }
        assert_eq!(
            output_path
                .with_extension(format!("{extension}.map"))
                .is_file(),
            expects_map
        );

        let metadata_path = output_path.with_extension(format!("{extension}.metadata.json"));
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(metadata_path).unwrap()).unwrap();
        assert_eq!(metadata["backend"], target);
        assert_eq!(metadata["compiler_version"], env!("CARGO_PKG_VERSION"));
    }
}
