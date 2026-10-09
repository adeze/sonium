#![no_main]

use libfuzzer_sys::fuzz_target;
use meshlang_ast::Declaration;
use meshlang_compiler::imports::ImportResolver;
use meshlang_compiler::{
    DiagLevel, GeoCodegen, PythonCodegen, TypeChecker, lower_to_canonical_ir, name_resolution,
};
use meshlang_parser::{ParseOptions, parse_with_options};
use std::path::Path;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > 1_048_576 {
        return;
    }
    let source = String::from_utf8_lossy(bytes);
    let Ok(ast) = parse_with_options(
        &source,
        "fuzz-input.mlg",
        ParseOptions {
            max_diagnostics: 16,
            max_nesting: 64,
            ..ParseOptions::default()
        },
    ) else {
        return;
    };

    // Exercise import-path validation without loading arbitrary files.
    let resolver = ImportResolver::new();
    for declaration in &ast.declarations {
        if let Declaration::Import(import) = declaration {
            let _ = resolver.resolve_path(&import.path, Path::new("/__meshlang_fuzz__/root.mlg"));
        }
    }

    if name_resolution::resolve(&ast).is_err() {
        return;
    }
    let mut checker = TypeChecker::new();
    if checker
        .check(&ast)
        .iter()
        .any(|diagnostic| diagnostic.level == DiagLevel::Error)
    {
        return;
    }
    let Ok(ir) = lower_to_canonical_ir(&ast) else {
        return;
    };

    let mut geo = GeoCodegen::new();
    if geo.apply_canonical_plan(&ir).is_ok() {
        let _ = geo.emit_source_file(&ast);
    }
    let mut python = PythonCodegen::new();
    if python.apply_canonical_plan(&ir).is_ok() {
        let _ = python.emit_source_file(&ast);
    }
});
