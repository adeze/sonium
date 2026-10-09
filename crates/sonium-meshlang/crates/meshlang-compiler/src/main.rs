//! MeshLang CLI compiler.
//!
//! Usage:
//!   meshlang compile input.mlg [-o output.geo] [--target geo|python]
//!   meshlang check input.mlg [--format text|json]
//!   meshlang lex input.mlg
//!   meshlang parse input.mlg

use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("MeshLang compiler v{}", env!("CARGO_PKG_VERSION"));
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  meshlang compile <input.mlg> [-o <output>] [--target geo|python]");
        eprintln!("  meshlang check   <input.mlg> [--format text|json]");
        eprintln!("  meshlang lex     <input.mlg>");
        eprintln!("  meshlang parse   <input.mlg>");
        process::exit(1);
    }

    let command = &args[1];
    let input_path = &args[2];

    let source = match fs::read_to_string(input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", input_path, e);
            process::exit(1);
        }
    };

    match command.as_str() {
        "lex" => cmd_lex(&source),
        "parse" => cmd_parse(&source, input_path),
        "check" => {
            let format = args
                .iter()
                .position(|a| a == "--format")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
                .unwrap_or("text");
            cmd_check(&source, input_path, format);
        }
        "compile" => {
            let target = args
                .iter()
                .position(|a| a == "--target")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
                .unwrap_or("geo");

            let output = args
                .iter()
                .position(|a| a == "-o")
                .and_then(|i| args.get(i + 1))
                .cloned();

            cmd_compile(&source, input_path, target, output);
        }
        other => {
            eprintln!("error: unknown command '{}'", other);
            process::exit(1);
        }
    }
}

fn cmd_lex(source: &str) {
    let tokens = meshlang_parser::lex(source);
    for tok in &tokens {
        println!(
            "{:?} {:?} @ {}..{}",
            tok.kind, tok.text, tok.span.start, tok.span.end
        );
    }
}

fn cmd_parse(source: &str, filename: &str) {
    match meshlang_parser::parse(source, filename) {
        Ok(ast) => {
            let document = meshlang_ast::VersionedAst::current(ast);
            let json = serde_json::to_string_pretty(&document).unwrap();
            println!("{}", json);
        }
        Err(e) => {
            eprintln!("{}", e.display_with_source(source, filename));
            process::exit(1);
        }
    }
}

fn cmd_check(source: &str, filename: &str, format: &str) {
    let ast = match meshlang_parser::parse(source, filename) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{}", e.display_with_source(source, filename));
            process::exit(1);
        }
    };

    let mut checker = meshlang_compiler::TypeChecker::new();
    let diagnostics = checker.check(&ast);

    match format {
        "json" => {
            let file_uri = format!(
                "file://{}",
                std::path::Path::new(filename)
                    .canonicalize()
                    .unwrap_or_else(|_| std::path::PathBuf::from(filename))
                    .display()
            );
            let json = meshlang_compiler::diagnostics::to_json(&diagnostics, source, &file_uri);
            println!("{}", json);
        }
        _ => {
            if diagnostics.is_empty() {
                eprintln!("check: {} — no issues", filename);
            } else {
                let formatted =
                    meshlang_compiler::diagnostics::format_terminal(&diagnostics, source, filename);
                eprint!("{}", formatted);

                let error_count = diagnostics
                    .iter()
                    .filter(|d| d.level == meshlang_compiler::DiagLevel::Error)
                    .count();
                let warn_count = diagnostics
                    .iter()
                    .filter(|d| d.level == meshlang_compiler::DiagLevel::Warning)
                    .count();

                if error_count > 0 {
                    eprintln!("check: {} error(s), {} warning(s)", error_count, warn_count);
                    process::exit(1);
                } else {
                    eprintln!("check: {} — {} warning(s)", filename, warn_count);
                }
            }
        }
    }
}

fn cmd_compile(source: &str, filename: &str, target: &str, output: Option<String>) {
    let result = match target {
        "geo" => meshlang_compiler::compile_to_geo_full(source, filename),
        "python" | "py" => meshlang_compiler::compile_to_python_full(source, filename),
        other => {
            eprintln!("error: unknown target '{}' (use 'geo' or 'python')", other);
            process::exit(1);
        }
    };

    match result {
        Ok(result) => {
            if !result.diagnostics.is_empty() {
                let formatted = meshlang_compiler::diagnostics::format_terminal(
                    &result.diagnostics,
                    source,
                    filename,
                );
                eprint!("{}", formatted);
            }

            if let Some(ref out_path) = output {
                write_file_or_exit(out_path, &result.code);
                eprintln!("wrote {}", out_path);
                write_artifact_sidecars(out_path, &result);
            } else {
                print!("{}", result.code);
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn write_artifact_sidecars(out_path: &str, result: &meshlang_compiler::CompileResult) {
    if let Some(map_json) = &result.tag_map_json {
        write_sidecar(out_path, "map", map_json);
    }
    write_sidecar(out_path, "ir.json", &result.canonical_ir_json);
    write_sidecar(
        out_path,
        "sensitivity.json",
        &result.sensitivity_manifest_json,
    );
    write_sidecar(
        out_path,
        "provenance.json",
        &result.provenance_manifest_json,
    );
    write_sidecar(out_path, "metadata.json", &result.backend_metadata_json);
}

fn write_sidecar(out_path: &str, suffix: &str, contents: &str) {
    let path = format!("{}.{}", out_path, suffix);
    write_file_or_exit(&path, contents);
    eprintln!("wrote {}", path);
}

fn write_file_or_exit(path: &str, contents: &str) {
    fs::write(path, contents).unwrap_or_else(|e| {
        eprintln!("error: cannot write '{}': {}", path, e);
        process::exit(1);
    });
}
