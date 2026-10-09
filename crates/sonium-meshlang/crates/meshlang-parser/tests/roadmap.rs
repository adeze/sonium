use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use meshlang_ast::span::Span;
use meshlang_ast::{
    AstNode, BinaryOp, CommentKind, Declaration, ElseBranch, Expr, MeshStatement,
    ProvenanceOperation, Statement, UnaryOp, VersionedAst,
};
use meshlang_parser::error::{DIAGNOSTIC_SCHEMA_VERSION, DiagnosticCode};
use meshlang_parser::lexer::TokenKind;
use meshlang_parser::parser::Parser;
use meshlang_parser::{
    GRAMMAR_VERSION, PARSE_JSON_SCHEMA_VERSION, ParseCache, ParseOptions, lex, parse, parse_json,
    parse_recovering, parse_with_options,
};

#[test]
fn lexer_and_tree_sitter_consume_the_normative_operator_contract() {
    for spelling in meshlang_parser::grammar::OPERATOR_SPELLINGS {
        let tokens = lex(spelling);
        let significant: Vec<_> = tokens
            .iter()
            .filter(|token| token.kind != TokenKind::Eof)
            .collect();
        assert_eq!(significant.len(), 1, "{spelling:?} must lex as one token");
        assert_eq!(significant[0].text(), *spelling);
    }

    let tree_sitter = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tree-sitter-meshlang/grammar.js"
    ));
    for spelling in meshlang_parser::grammar::OPERATOR_SPELLINGS {
        assert!(
            tree_sitter.contains(&format!("'{spelling}'")),
            "tree-sitter grammar is missing operator spelling {spelling:?}"
        );
    }
}

#[test]
fn shipping_meshlang_product_corpus_parses_deterministically() {
    // Keep this corpus at the parser boundary: parsing the examples must not
    // start a compiler, mesher, or simulator.  The same source is parsed
    // twice so a future lexer/parser cache or recovery change cannot silently
    // make product fixtures order-dependent.
    let corpus = [
        (
            "examples/gmsh_helmholtz_resonator.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/gmsh_helmholtz_resonator.mlg"
            )),
        ),
        (
            "examples/gmsh_t16_csg.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/gmsh_t16_csg.mlg"
            )),
        ),
        (
            "examples/gmsh_t19_loft_pipe.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/gmsh_t19_loft_pipe.mlg"
            )),
        ),
        (
            "examples/gmsh_t5_cheese.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/gmsh_t5_cheese.mlg"
            )),
        ),
        (
            "examples/ported_acoustic_frame.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/ported_acoustic_frame.mlg"
            )),
        ),
        (
            "examples/room_acoustics.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/room_acoustics.mlg"
            )),
        ),
        (
            "examples/speaker_cabinet.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/speaker_cabinet.mlg"
            )),
        ),
        (
            "examples/sphere_scattering.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/sphere_scattering.mlg"
            )),
        ),
        (
            "stdlib/acoustics.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../stdlib/acoustics.mlg"
            )),
        ),
        (
            "stdlib/shapes.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../stdlib/shapes.mlg"
            )),
        ),
        (
            "sonium-sim-pcfan/shapes/pcfan.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../sonium-sim-pcfan/shapes/pcfan.mlg"
            )),
        ),
        (
            "sonium-sim-speaker/examples/two_way_speaker.mlg",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../sonium-sim-speaker/examples/two_way_speaker.mlg"
            )),
        ),
    ];

    for (name, source) in corpus {
        let first = parse(source, name)
            .unwrap_or_else(|error| panic!("shipping fixture {name} should parse: {error}"));
        let second = parse(source, name)
            .unwrap_or_else(|error| panic!("shipping fixture {name} should reparse: {error}"));
        let first_json = serde_json::to_string(&first).expect("first AST should serialize");
        let second_json = serde_json::to_string(&second).expect("second AST should serialize");
        assert_eq!(
            first_json, second_json,
            "shipping fixture {name} is not deterministic"
        );
        assert!(
            !first.declarations.is_empty(),
            "shipping fixture {name} unexpectedly has no declarations"
        );
    }
}

#[test]
fn mixed_geometry_boolean_and_pipeline_operators_use_longest_match() {
    let tokens = lex("a | b |> trim() || c && d ** 2");
    let kinds: Vec<_> = tokens.iter().map(|token| token.kind).collect();
    let pipe = kinds
        .iter()
        .position(|kind| *kind == TokenKind::Pipe)
        .unwrap();
    assert_eq!(kinds[pipe + 1], TokenKind::Ident);
    assert!(kinds.contains(&TokenKind::PipeArrow));
    assert!(kinds.contains(&TokenKind::PipePipe));
    assert!(kinds.contains(&TokenKind::AmpAmp));
    assert!(kinds.contains(&TokenKind::StarStar));
    assert!(!kinds.iter().any(|kind| {
        *kind == TokenKind::Pipe
            && kinds.get(
                kinds
                    .iter()
                    .position(|candidate| candidate == kind)
                    .unwrap()
                    + 1,
            ) == Some(&TokenKind::Gt)
    }));
}

#[test]
fn parses_every_optimization_and_physics_declaration() {
    let source = r#"
version 1
design thickness {
    type: continuous,
    initial: 1.0,
    lower: 0.1,
    upper: 2.0,
    unit: "m",
    distribution: normal(mean: 1.0, standard_deviation: 0.1)
}
uncertainty load { distribution: uniform(lower: 0.0, upper: 1.0) }
objective compliance { sense: minimize, expression: strain_energy, weight: 1.0 }
constraint volume { kind: volume_fraction, region: domain, maximum: 0.5 }
design_region topology_domain {
    kind: design,
    region: domain,
    annotations: [
        minimum_thickness(value: 0.1, unit: "m"),
        minimum_radius(value: 0.2, unit: "m"),
        overhang(maximum_angle: 45.0, direction: [0.0, 0.0, 1.0]),
        symmetry(kind: plane(origin: [0.0, 0.0, 0.0], normal: [1.0, 0.0, 0.0])),
        draw_direction(direction: [0.0, 0.0, 1.0]),
        frozen
    ]
}
physical_region volume solid { entities: body, material: Steel }
boundary surface walls { entities: wall_faces, kind: dirichlet, value: 0.0 }
interface surface coupling { primary: fluid_side, secondary: solid_side, kind: mortar, order: 2 }
coordinate_system local {
    kind: cartesian,
    origin: [0.0, 0.0, 0.0],
    axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
material_region alloy { region: body, material: Steel, density: 7800.0 }
field density { kind: topology, region: domain, initial_density: 0.5 }
mesh {
    metric domain { kind: isotropic, size: 0.01 }
}
"#;

    let file = parse(source, "metadata.mlg").expect("metadata grammar must parse");
    assert_eq!(file.declarations.len(), 12);
    assert!(matches!(
        file.declarations[0],
        Declaration::DesignVariable(_)
    ));
    assert!(matches!(file.declarations[1], Declaration::Uncertainty(_)));
    assert!(matches!(file.declarations[2], Declaration::Objective(_)));
    assert!(matches!(file.declarations[3], Declaration::Constraint(_)));
    assert!(matches!(file.declarations[4], Declaration::DesignRegion(_)));
    assert!(matches!(
        file.declarations[5],
        Declaration::PhysicalRegion(_)
    ));
    assert!(matches!(file.declarations[6], Declaration::Boundary(_)));
    assert!(matches!(file.declarations[7], Declaration::Interface(_)));
    assert!(matches!(
        file.declarations[8],
        Declaration::CoordinateSystem(_)
    ));
    assert!(matches!(
        file.declarations[9],
        Declaration::MaterialRegion(_)
    ));
    assert!(matches!(
        file.declarations[10],
        Declaration::GeometryField(_)
    ));
    let Declaration::Mesh(mesh) = &file.declarations[11] else {
        panic!("expected mesh declaration");
    };
    assert!(matches!(mesh.statements[0], MeshStatement::Metric(_)));
}

#[test]
fn new_modelling_words_remain_contextual_identifiers() {
    let source = r#"
geometry compatibility {
    let boundary = 1
    let interface = boundary + 1
    let design = interface + 1
    let metric = design + 1
    metric
}
"#;
    parse(source, "contextual.mlg").expect("new top-level words must remain local identifiers");
}

#[test]
fn recovering_parse_reports_multiple_errors_and_keeps_later_declarations() {
    let source =
        "param bad_number = 1e+\nconst first = 1\nparam bad_string = \"oops\nconst second = 2\n";
    let report = parse_recovering(
        source,
        "recovery.mlg",
        ParseOptions {
            file_id: 41,
            max_diagnostics: 8,
            max_nesting: 32,
            ..ParseOptions::default()
        },
    );
    assert_eq!(report.diagnostics.len(), 2);
    assert!(
        report
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.code == DiagnosticCode::Lexical)
    );
    assert!(
        report
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.span.file_id == 41)
    );
    assert_eq!(report.source_file.declarations.len(), 2);
}

#[test]
fn recovery_does_not_resynchronize_on_nested_declaration_keywords() {
    let source = "geometry broken { let missing = param nested = 1 } param good = 2";
    let report = parse_recovering(source, "nested-recovery.mlg", ParseOptions::default());

    assert!(!report.diagnostics.is_empty());
    assert!(report.source_file.declarations.iter().any(|declaration| {
        matches!(declaration, Declaration::Param(parameter) if parameter.name.name == "good")
    }));
    assert!(!report.source_file.declarations.iter().any(|declaration| {
        matches!(declaration, Declaration::Param(parameter) if parameter.name.name == "nested")
    }));
}

#[test]
fn comment_spans_targets_and_source_identity_survive_versioned_json() {
    let source = "/// documented parameter\n/** documented block */\n// ordinary line\n/* ordinary block */\nparam radius = 1.0 m\n";
    let file = parse(source, "comments.mlg").expect("comment fixture must parse");
    let original_identity = file.metadata.identity.clone();
    let bytes = VersionedAst::current(file)
        .to_json_vec()
        .expect("versioned AST must serialize");
    let (decoded, migration) =
        VersionedAst::from_json_slice(&bytes).expect("versioned AST must deserialize");
    assert!(!migration.was_migrated());
    let comments = decoded.source_file.comments;
    assert_eq!(comments.len(), 4);
    assert_eq!(comments[0].kind, CommentKind::DocLine);
    assert_eq!(comments[0].text, "/// documented parameter");
    assert!(comments[0].target.is_some());
    assert_eq!(comments[1].kind, CommentKind::DocBlock);
    assert_eq!(comments[1].text, "/** documented block */");
    assert!(comments[1].target.is_some());
    assert_eq!(comments[2].kind, CommentKind::Line);
    assert_eq!(comments[2].text, "// ordinary line");
    assert!(comments[2].target.is_none());
    assert_eq!(comments[3].kind, CommentKind::Block);
    assert_eq!(comments[3].text, "/* ordinary block */");
    assert!(comments[3].target.is_none());
    assert_eq!(comments[0].span.file_id, 0);
    assert_eq!(decoded.source_file.metadata.identity, original_identity);
    assert!(decoded.source_file.metadata.identity.content_hash.is_some());
}

#[test]
fn doc_comments_are_lossless_trivia_inside_blocks() {
    let source = "geometry sample {\n  /// local shape\n  let shape = Box(1, 1, 1)\n  shape\n}\nparam next = 1\n";
    let file = parse(source, "local-doc.mlg").unwrap();
    assert!(matches!(file.declarations[0], Declaration::Geometry(_)));
    assert_eq!(file.comments.len(), 1);
    assert_eq!(file.comments[0].kind, CommentKind::DocLine);
    assert_eq!(file.comments[0].text, "/// local shape");
    assert_eq!(file.comments[0].target, None);
    assert_eq!(
        &source[file.comments[0].span.start..file.comments[0].span.end],
        "/// local shape"
    );
}

#[test]
fn typed_geometry_ports_parse_and_old_ast_payload_defaults() {
    let source = r#"
        geometry frame(width: Float = 1.0) -> (body: Solid, inlet: Surface) {
            let body = Box(width, 1.0, 1.0)
            let inlet = Rectangle(1.0, 1.0)
        }
    "#;
    let file = parse(source, "ports.mlg").expect("typed geometry ports must parse");
    let Declaration::Geometry(geometry) = &file.declarations[0] else {
        panic!("expected geometry declaration");
    };
    assert_eq!(geometry.ports.len(), 2);
    assert_eq!(geometry.ports[0].name.name, "body");
    assert_eq!(geometry.ports[1].name.name, "inlet");

    let encoded = VersionedAst::current(file.clone())
        .to_json_vec()
        .expect("ports AST must serialize");
    let (decoded, migration) =
        VersionedAst::from_json_slice(&encoded).expect("ports AST must deserialize");
    assert!(!migration.was_migrated());
    let Declaration::Geometry(decoded_geometry) = &decoded.source_file.declarations[0] else {
        panic!("expected decoded geometry declaration");
    };
    assert_eq!(decoded_geometry.ports.len(), 2);

    let mut legacy: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    legacy["source_file"]["declarations"][0]["Geometry"]
        .as_object_mut()
        .expect("geometry envelope")
        .remove("ports");
    let (decoded_legacy, migration) =
        VersionedAst::from_json_slice(&serde_json::to_vec(&legacy).expect("legacy AST JSON"))
            .expect("AST without ports must remain compatible");
    assert!(!migration.was_migrated());
    let Declaration::Geometry(legacy_geometry) = &decoded_legacy.source_file.declarations[0] else {
        panic!("expected legacy geometry declaration");
    };
    assert!(legacy_geometry.ports.is_empty());
}

#[test]
fn arbitrary_bytes_and_deep_nesting_return_without_panicking() {
    let mut state = 0x9e37_79b9_u32;
    for length in 0..512 {
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((state >> 24) as u8);
        }
        let source = String::from_utf8_lossy(&bytes);
        let result = catch_unwind(AssertUnwindSafe(|| {
            parse_recovering(&source, "fuzz-bytes.mlg", ParseOptions::default())
        }));
        assert!(
            result.is_ok(),
            "panic for generated input of length {length}"
        );
    }

    let nested = format!("param deep = {}1{}", "(".repeat(10_000), ")".repeat(10_000));
    let report = parse_recovering(
        &nested,
        "deep.mlg",
        ParseOptions {
            max_nesting: 32,
            ..ParseOptions::default()
        },
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::NestingLimit)
    );
}

#[test]
fn arbitrary_valid_utf8_recovery_keeps_diagnostics_spans_and_metadata_bounded() {
    // Deterministic valid UTF-8 fuzzing complements the byte corpus above: every
    // generated input can exercise lexer/parser character-boundary handling.
    let alphabet: &[char] = &[
        'a', 'Z', '0', '_', ' ', '\n', '\r', '(', ')', '[', ']', '{', '}', ',', ':', ';', '=', '+',
        '-', '*', '/', '|', '&', '"', '\'', 'é', 'λ', '中', '🚀', '\u{2003}', '\u{0301}',
    ];
    let options = ParseOptions {
        file_id: 73,
        max_diagnostics: 7,
        max_nesting: 24,
        ..ParseOptions::default()
    };
    let mut state = 0xd1b5_4a32_u32;

    for character_count in 0..256 {
        let mut source = String::with_capacity(character_count * 4);
        for _ in 0..character_count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            source.push(alphabet[(state as usize) % alphabet.len()]);
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            parse_recovering(&source, "fuzz-utf8.mlg", options)
        }));
        let report = result.unwrap_or_else(|_| {
            panic!("panic for valid UTF-8 input with {character_count} characters")
        });

        assert!(report.diagnostics.len() <= options.max_diagnostics);
        assert_eq!(
            report.source_file.metadata.identity.display_name,
            "fuzz-utf8.mlg"
        );
        assert!(report.source_file.metadata.identity.content_hash.is_some());
        assert!(report.source_file.structural_hash().is_ok());

        let assert_span = |span: Span| {
            assert_eq!(span.file_id, options.file_id);
            assert!(span.start <= span.end && span.end <= source.len());
            assert!(source.is_char_boundary(span.start));
            assert!(source.is_char_boundary(span.end));
        };
        assert_span(report.source_file.span());
        for diagnostic in &report.diagnostics {
            assert_span(diagnostic.span);
        }
        for declaration in &report.source_file.declarations {
            assert_span(declaration.span());
            let _ = declaration.node_id(&report.source_file.metadata.identity);
        }
        for comment in &report.source_file.comments {
            assert_span(comment.span);
        }
    }
}

#[test]
fn file_cache_uses_content_and_options() {
    let mut cache = ParseCache::default();
    let options = ParseOptions::default();
    let first = cache.parse("param x = 1", "cache.mlg", options);
    let second = cache.parse("param x = 1", "cache.mlg", options);
    assert_eq!(
        first.source_file.structural_hash().unwrap(),
        second.source_file.structural_hash().unwrap()
    );
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 1);
    cache.parse("param x = 2", "cache.mlg", options);
    assert_eq!(cache.stats().misses, 2);
}

#[test]
fn normative_precedence_and_pipe_tokenization_match_grammar_1_0() {
    assert_eq!(GRAMMAR_VERSION, "1.0");
    assert_eq!(DIAGNOSTIC_SCHEMA_VERSION, 2);
    let tokens = lex("a | b || c |> transform()");
    assert_eq!(tokens[1].kind, TokenKind::Pipe);
    assert_eq!(tokens[3].kind, TokenKind::PipePipe);
    assert_eq!(tokens[5].kind, TokenKind::PipeArrow);

    let file = parse(
        "param value = -2 ** 3 ** 2 + 4 * 5\nparam shape = a | b && c |> transform()",
        "precedence.mlg",
    )
    .unwrap();
    let Declaration::Param(value) = &file.declarations[0] else {
        panic!("expected parameter");
    };
    let Expr::Binary(add) = &value.value else {
        panic!("additive expression must be the root");
    };
    assert_eq!(add.op, BinaryOp::Add);
    let Expr::Unary(negate) = add.left.as_ref() else {
        panic!("unary negation must bind outside power");
    };
    assert_eq!(negate.op, UnaryOp::Neg);
    let Expr::Binary(power) = negate.operand.as_ref() else {
        panic!("expected power under negation");
    };
    assert_eq!(power.op, BinaryOp::Pow);
    assert!(matches!(power.right.as_ref(), Expr::Binary(right) if right.op == BinaryOp::Pow));
    assert!(matches!(add.right.as_ref(), Expr::Binary(right) if right.op == BinaryOp::Mul));

    let Declaration::Param(shape) = &file.declarations[1] else {
        panic!("expected shape parameter");
    };
    let Expr::Pipe(pipe) = &shape.value else {
        panic!("pipe arrow must have lowest precedence");
    };
    assert!(matches!(pipe.value.as_ref(), Expr::Binary(union) if union.op == BinaryOp::CsgUnion));
}

#[test]
fn precedence_rows_drive_parser_ast_shape_contract() {
    type ShapeCheck = fn(&Expr) -> bool;

    let cases: &[(&str, &str, ShapeCheck)] = &[
        ("pipe", "a |> transform()", |expr| {
            matches!(expr, Expr::Pipe(_))
        }),
        (
            "csg",
            "a | b",
            |expr| matches!(expr, Expr::Binary(binary) if binary.op == BinaryOp::CsgUnion),
        ),
        (
            "or",
            "true || false",
            |expr| matches!(expr, Expr::Binary(binary) if binary.op == BinaryOp::Or),
        ),
        (
            "and",
            "true && false",
            |expr| matches!(expr, Expr::Binary(binary) if binary.op == BinaryOp::And),
        ),
        (
            "comparison",
            "1 < 2",
            |expr| matches!(expr, Expr::Binary(binary) if binary.op == BinaryOp::Lt),
        ),
        (
            "additive",
            "1 + 2",
            |expr| matches!(expr, Expr::Binary(binary) if binary.op == BinaryOp::Add),
        ),
        (
            "multiplicative",
            "1 * 2",
            |expr| matches!(expr, Expr::Binary(binary) if binary.op == BinaryOp::Mul),
        ),
        (
            "unary",
            "-1",
            |expr| matches!(expr, Expr::Unary(unary) if unary.op == UnaryOp::Neg),
        ),
        (
            "power",
            "2 ** 3",
            |expr| matches!(expr, Expr::Binary(binary) if binary.op == BinaryOp::Pow),
        ),
        ("postfix", "value.field", |expr| {
            matches!(expr, Expr::FieldAccess(_))
        }),
        ("primary", "1", |expr| matches!(expr, Expr::Literal(_))),
    ];

    let rows = meshlang_parser::grammar::PRECEDENCE_ROWS;
    assert_eq!(
        rows.len(),
        cases.len(),
        "every precedence row needs a parser case"
    );
    for ((row, source, check), (contract_name, _, right_associative)) in
        cases.iter().zip(rows.iter())
    {
        assert_eq!(
            row, contract_name,
            "parser case drifted from PRECEDENCE_ROWS"
        );
        let file = parse(&format!("param value = {source}"), "precedence-row.mlg")
            .unwrap_or_else(|error| panic!("{row} fixture should parse: {error}"));
        let Declaration::Param(parameter) = &file.declarations[0] else {
            panic!("{row} fixture did not produce a parameter");
        };
        assert!(
            check(&parameter.value),
            "{row} fixture has the wrong AST shape"
        );
        if *right_associative {
            assert_eq!(*row, "power");
            let nested = parse("param value = 2 ** 3 ** 4", "precedence-row.mlg")
                .expect("right-associative power fixture should parse");
            let Declaration::Param(nested_parameter) = &nested.declarations[0] else {
                panic!("power associativity fixture did not produce a parameter");
            };
            assert!(matches!(
                &nested_parameter.value,
                Expr::Binary(binary)
                    if binary.op == BinaryOp::Pow
                        && matches!(binary.right.as_ref(), Expr::Binary(right) if right.op == BinaryOp::Pow)
            ));
        }
    }
}

#[test]
fn composite_expression_spans_cover_the_complete_expression() {
    let source = "param value = base.part[2] + rhs |> translate(1)";
    let file = parse(source, "spans.mlg").unwrap();
    let Declaration::Param(parameter) = &file.declarations[0] else {
        panic!("expected parameter");
    };
    let Expr::Pipe(pipe) = &parameter.value else {
        panic!("expected pipe expression");
    };
    assert_eq!(
        &source[pipe.span.start..pipe.span.end],
        "base.part[2] + rhs |> translate(1)"
    );

    let Expr::Binary(add) = pipe.value.as_ref() else {
        panic!("expected additive expression");
    };
    assert_eq!(&source[add.span.start..add.span.end], "base.part[2] + rhs");

    let Expr::Index(index) = add.left.as_ref() else {
        panic!("expected index expression");
    };
    assert_eq!(&source[index.span.start..index.span.end], "base.part[2]");

    let Expr::FieldAccess(field) = index.object.as_ref() else {
        panic!("expected field access expression");
    };
    assert_eq!(&source[field.span.start..field.span.end], "base.part");
}

#[test]
fn parsed_ast_records_source_provenance() {
    let source = "param width = 1.0 m\ngeometry domain { let body = Box(1.0, 1.0, 1.0) }\n";
    let file = parse(source, "provenance.mlg").expect("source should parse");

    assert_eq!(file.provenance.records.len(), file.declarations.len() + 1);
    assert!(
        file.provenance
            .records
            .iter()
            .all(|record| record.operation == ProvenanceOperation::Parsed)
    );
    for declaration in &file.declarations {
        let node = declaration.node_id(&file.metadata.identity);
        assert!(file.provenance.for_node(node).next().is_some());
    }
}

#[test]
fn projected_geometry_field_parses_and_rejects_unknown_methods() {
    let source = "field surface { kind: projected, geometry: domain, source_field: pressure, method: iso_surface, tolerance: 0.01 }";
    let file = parse(source, "projected-field.mlg").expect("projected field should parse");
    let Declaration::GeometryField(field) = &file.declarations[0] else {
        panic!("expected geometry field declaration");
    };
    let meshlang_ast::GeometryFieldKind::Projected { provenance, .. } = &field.kind else {
        panic!("expected projected geometry field");
    };
    assert_eq!(provenance.source_field.name, "pressure");
    assert_eq!(
        provenance.method,
        meshlang_ast::ProjectionMethod::IsoSurface
    );
    assert!(provenance.tolerance.is_some());
    let error = parse(
        "field surface { kind: projected, geometry: domain, source_field: pressure, method: unsupported }",
        "projected-field-invalid.mlg",
    )
    .expect_err("unknown projection method must be rejected");
    assert_eq!(error.code, DiagnosticCode::UnexpectedToken);
}

#[test]
fn integer_overflow_is_reported_as_invalid_number() {
    let error = parse(
        "param count = 999999999999999999999999999999999999",
        "integer-overflow.mlg",
    )
    .expect_err("integer overflow must not wrap");
    assert_eq!(error.code, DiagnosticCode::InvalidNumber);
}

#[test]
fn contextual_from_keyword_is_valid_line_constructor_argument() {
    let source = r#"
        geometry model {
            let start = Point(0.0, 0.0, 0.0)
            let finish = Point(1.0, 0.0, 0.0)
            let edge = Line(from: start, to: finish)
        }
    "#;
    parse(source, "line-from-argument.mlg").expect("Line(from:, to:) should parse");
}

#[test]
fn geometry_constructor_calls_participate_in_csg_and_pipeline_parsing() {
    let source = r#"
geometry model {
    let shape = Box(1.0, 1.0, 1.0) | Cylinder(radius: 0.25, height: 1.0) |> transform()
}
"#;
    parse(source, "geometry-calls.mlg").expect("constructor CSG/pipeline expression should parse");
}

#[test]
fn version_numeric_string_unit_and_diagnostic_limits_are_explicit() {
    let unsupported = parse("version 2\nparam x = 1", "version.mlg").unwrap_err();
    assert_eq!(unsupported.code, DiagnosticCode::UnsupportedVersion);
    assert_eq!(
        parse("param x = 1e999", "overflow.mlg").unwrap_err().code,
        DiagnosticCode::InvalidNumber
    );
    assert_eq!(
        parse("param x = 1.0 made_up_unit", "unit.mlg")
            .unwrap_err()
            .code,
        DiagnosticCode::InvalidUnit
    );
    let parsed = parse(r#"param text = "line\nquote: \"""#, "string.mlg").unwrap();
    assert_eq!(parsed.declarations.len(), 1);

    let percentage = parse("param ratio = 50 %", "percentage.mlg").unwrap();
    let Declaration::Param(ratio) = &percentage.declarations[0] else {
        panic!("expected percentage parameter");
    };
    assert_eq!(
        ratio.unit.as_ref().map(|unit| unit.text.as_str()),
        Some("%")
    );

    let modulo = parse("param remainder = 50 % 7", "modulo.mlg").unwrap();
    let Declaration::Param(remainder) = &modulo.declarations[0] else {
        panic!("expected modulo parameter");
    };
    assert!(matches!(
        remainder.value,
        Expr::Binary(ref expression) if expression.op == BinaryOp::Mod
    ));

    let report = parse_recovering(
        "? const a = 1 ? const b = 2 ? const c = 3 ? const d = 4",
        "bounded.mlg",
        ParseOptions {
            max_diagnostics: 3,
            ..ParseOptions::default()
        },
    );
    assert!(report.diagnostics.len() <= 3);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::TooManyDiagnostics)
    );

    // A zero limit used to record no diagnostics at all and report success
    // on erroneous input. It now clamps to one truncation diagnostic so the
    // failure is never silent: parsing stops at the first error and the
    // truncation notice documents that the remaining input — including the
    // good declaration — was not parsed.
    let zero_limit = parse_recovering(
        "? const a = 1 ? const b = 2 param good = 3",
        "zero-limit.mlg",
        ParseOptions {
            max_diagnostics: 0,
            ..ParseOptions::default()
        },
    );
    assert!(!zero_limit.is_success());
    assert!(
        zero_limit
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::TooManyDiagnostics)
    );
    assert!(zero_limit
        .source_file
        .declarations
        .iter()
        .all(|declaration| {
            !matches!(declaration, Declaration::Param(parameter) if parameter.name.name == "good")
        }));
}

#[test]
fn source_size_limit_is_typed_and_applies_to_strict_and_recovering_reads() {
    let source = "param x = 1\n";
    let options = ParseOptions {
        max_source_bytes: source.len() - 1,
        ..ParseOptions::default()
    };

    let strict = parse_with_options(source, "too-large.mlg", options).unwrap_err();
    assert_eq!(strict.code, DiagnosticCode::SourceTooLarge);
    assert_eq!(strict.span.start, 0);
    assert_eq!(strict.span.end, source.len());
    assert_eq!(strict.source_name.as_deref(), Some("too-large.mlg"));

    let report = parse_recovering(source, "too-large.mlg", options);
    assert_eq!(report.diagnostics.len(), 1);
    assert_eq!(report.diagnostics[0].code, DiagnosticCode::SourceTooLarge);
    assert!(report.source_file.declarations.is_empty());
    assert_eq!(
        report.source_file.metadata.identity.display_name,
        "too-large.mlg"
    );
}

#[test]
fn tooling_parse_json_is_versioned_and_keeps_typed_diagnostics_with_ast() {
    let json: serde_json::Value = parse_json(
        "param broken = 1e+\nparam good = 2\n",
        "tooling.mlg",
        ParseOptions::default(),
    )
    .unwrap()
    .parse()
    .unwrap();
    assert_eq!(json["parse_schema"], PARSE_JSON_SCHEMA_VERSION);
    assert_eq!(json["grammar_version"], GRAMMAR_VERSION);
    assert_eq!(json["diagnostic_schema"], DIAGNOSTIC_SCHEMA_VERSION);
    assert_eq!(json["report"]["diagnostics"][0]["code"], "lexical");
    assert!(json["report"]["source_file"]["metadata"].is_object());
}

#[test]
fn trailing_separators_are_canonical_and_empty_elements_are_rejected() {
    parse(
        r#"
import { Shape, Material, } from "library.mlg"
geometry separator(a: Float,) {
    let values = [1, 2,]
    Box(1, 2,)
}

mesh {
    compound surface(a, b,)
    transfinite curve(edge, nodes: 4,)
    refine field(size: 1.0,) { value: 1.0 }
}
"#,
        "separators.mlg",
    )
    .expect("documented trailing separators must parse");

    for invalid in ["param x = f(,1)", "param x = [1,,2]"] {
        assert!(
            parse(invalid, "invalid-separator.mlg").is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn empty_mesh_compound_is_rejected_with_a_typed_parser_diagnostic() {
    let error = parse(
        r#"
            geometry model {
                let body = Box(1.0, 1.0, 1.0)
            }
            mesh model {
                compound surface()
            }
        "#,
        "empty-compound.mlg",
    )
    .expect_err("empty mesh compounds must not parse as an implicit entity");
    assert_eq!(error.code, DiagnosticCode::UnexpectedToken);
    assert!(error.message.contains("expected expression"), "{error}");
}

#[test]
fn named_mesh_blocks_accept_both_documented_target_spellings() {
    for (spelling, filename) in [
        ("mesh model {}", "mesh-named.mlg"),
        ("mesh for model {}", "mesh-for.mlg"),
    ] {
        let source = format!("geometry model {{ Box(1.0, 1.0, 1.0) }}\n{spelling}");
        let file = parse(&source, filename).expect("named mesh syntax must parse");
        let Declaration::Mesh(mesh) = &file.declarations[1] else {
            panic!("expected mesh declaration");
        };
        assert_eq!(
            mesh.target.as_ref().map(|target| target.name.as_str()),
            Some("model")
        );
    }
}

#[test]
fn all_shipping_mlg_files_parse_in_the_smoke_tier() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let files = [
        "../../examples/gmsh_helmholtz_resonator.mlg",
        "../../examples/gmsh_t16_csg.mlg",
        "../../examples/gmsh_t19_loft_pipe.mlg",
        "../../examples/gmsh_t5_cheese.mlg",
        "../../examples/room_acoustics.mlg",
        "../../examples/speaker_cabinet.mlg",
        "../../examples/sphere_scattering.mlg",
        "../../stdlib/acoustics.mlg",
        "../../stdlib/shapes.mlg",
        "../../../sonium-sim-pcfan/shapes/pcfan.mlg",
        "../../../sonium-sim-speaker/examples/two_way_speaker.mlg",
    ];
    for relative in files {
        let path = manifest.join(relative);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        parse(&source, &path.to_string_lossy())
            .unwrap_or_else(|error| panic!("{}: {} at {:?}", path.display(), error, error.span));
    }
}

#[test]
fn permanent_fuzz_regression_corpus_stays_bounded() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for fixture in [
        "fuzz/corpus/parse_recovering/empty-kv-args.mlg",
        "fuzz/corpus/parse_recovering/unterminated-string.mlg",
        "fuzz/corpus/parse_recovering/nested-comment.mlg",
        "fuzz/corpus/parse_recovering/recovery.mlg",
        "fuzz/corpus/parse_recovering/comparison-span.mlg",
        "fuzz/corpus/parse_recovering/pipe-arrow.mlg",
    ] {
        let path = manifest.join(fixture);
        let bytes = std::fs::read(&path).unwrap();
        let source = String::from_utf8_lossy(&bytes);
        let report = parse_recovering(&source, &path.to_string_lossy(), ParseOptions::default());
        assert!(report.diagnostics.len() <= ParseOptions::default().max_diagnostics);
    }
}

#[test]
fn recovered_diagnostics_match_the_stable_json_snapshot() {
    let report = parse_recovering(
        "@ const recovered = 1\n#",
        "snapshot.mlg",
        ParseOptions {
            file_id: 7,
            ..ParseOptions::default()
        },
    );
    assert_eq!(
        serde_json::to_value(&report.diagnostics).unwrap(),
        serde_json::json!([
            {
                "code": "lexical",
                "message": "unexpected character",
                "span": { "start": 0, "end": 1, "file_id": 7 },
                "source_name": "snapshot.mlg"
            },
            {
                "code": "lexical",
                "message": "unexpected character",
                "span": { "start": 22, "end": 23, "file_id": 7 },
                "source_name": "snapshot.mlg"
            }
        ])
    );
}

#[test]
fn newline_filename_and_empty_token_behavior_is_platform_independent() {
    let lf = parse("param café = 1\nconst value = 2\n", r"C:\models\café.mlg").unwrap();
    let crlf = parse(
        "param café = 1\r\nconst value = 2\r\n",
        r"C:\models\café.mlg",
    )
    .unwrap();
    assert_eq!(lf.declarations.len(), crlf.declarations.len());
    assert_eq!(lf.metadata.identity.display_name, r"C:\models\café.mlg");

    let empty = Parser::new(Vec::new(), 99)
        .parse_source_file()
        .expect("an externally supplied empty token stream must be total");
    assert!(empty.declarations.is_empty());
    assert_eq!(empty.span.file_id, 99);
}

#[test]
fn unicode_diagnostics_render_scalar_width_not_byte_width() {
    let error = parse("🚀", "unicode.mlg").unwrap_err();
    let rendered = error.display_with_source("🚀", "unicode.mlg");
    assert!(rendered.ends_with("   | ^\n"), "{rendered}");
}

#[test]
fn parser_retains_dynamic_else_if_chain_structure() {
    let source = r#"
        param enabled = true
        geometry model {
            if enabled {
                let primary = Box(1.0, 1.0, 1.0)
                let primary_copy = primary |> translate(1.0, 0.0, 0.0)
            } else if !enabled {
                let fallback = Box(0.5, 0.5, 0.5)
                let fallback_copy = fallback |> translate(-1.0, 0.0, 0.0)
            } else {
                let final_shape = Box(0.25, 0.25, 0.25)
            }
        }
    "#;
    let file = parse(source, "dynamic-else-if.mlg").expect("else-if source should parse");
    let Declaration::Geometry(geometry) = &file.declarations[1] else {
        panic!("expected geometry declaration");
    };
    let Statement::If(statement) = &geometry.body.statements[0] else {
        panic!("expected geometry if statement");
    };
    let Some(ElseBranch::ElseIf(next)) = &statement.else_block else {
        panic!("expected else-if branch");
    };
    assert!(matches!(next.else_block, Some(ElseBranch::Else(_))));
    assert_eq!(statement.then_block.statements.len(), 2);
    assert_eq!(next.then_block.statements.len(), 2);
}
