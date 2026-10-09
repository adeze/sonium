//! Machine-readable MeshLang grammar contract shared by parser tooling.
//!
//! The complete prose contract lives at `crates/sonium-meshlang/GRAMMAR.md`.
//! The tree-sitter grammar is validated against the same rows so editor
//! parsing cannot silently drift from the Rust parser.

use thiserror::Error;

/// Longest-match operator spellings, ordered from longest to shortest.
///
/// Covers every multi-character operator the lexer emits (`|>`, `||`, `&&`,
/// `**`, `==`, `!=`, `<=`, `>=`, `->`, `..`, `..=`) plus the single-character
/// spellings they prefix-collide with (`|`, `&`, `-`, `<`, `>`, `!`), so
/// longest-match drift in any of them trips the grammar contract.
pub const OPERATOR_SPELLINGS: &[&str] = &[
    "|>", "..=", "||", "&&", "**", "==", "!=", "<=", ">=", "->", "..", "|", "&", "-", "<", ">", "!",
];

/// Expression precedence rows, from lowest to highest.
///
/// `..` and `..=` are range delimiters for `for` clauses, not expression
/// operators, so they deliberately do not appear in this table.
pub const PRECEDENCE_ROWS: &[(&str, u8, bool)] = &[
    ("pipe", 1, false),
    ("csg", 2, false),
    ("or", 3, false),
    ("and", 4, false),
    ("comparison", 5, false),
    ("additive", 6, false),
    ("multiplicative", 7, false),
    ("unary", 8, false),
    ("power", 9, true),
    ("postfix", 10, false),
    ("primary", 11, false),
];

/// Tree-sitter numeric precedence values for the same expression rows.
///
/// Tree-sitter leaves the pipeline production unwrapped and starts numeric
/// precedence at the CSG row. Numeric values are separate from
/// [`PRECEDENCE_ROWS`] because only their ordering is normative.
const TREE_SITTER_PRECEDENCE: &[(&str, u8)] = &[
    ("csg_expr", 3),
    ("or_expr", 4),
    ("and_expr", 5),
    ("comparison_expr", 6),
    ("additive_expr", 7),
    ("multiplicative_expr", 8),
    ("unary_expr", 9),
    ("power_expr", 10),
    ("postfix_expr", 11),
    ("constructor_expr", 12),
];

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GrammarContractError {
    #[error("tree-sitter grammar is missing rule {rule:?}")]
    MissingRule { rule: String },
    #[error("tree-sitter grammar rule {rule:?} is missing precedence {precedence}")]
    MissingPrecedence { rule: String, precedence: u8 },
    #[error("tree-sitter grammar rule {rule:?} has unexpected precedence values {values:?}")]
    UnexpectedPrecedence { rule: String, values: Vec<u8> },
    #[error("tree-sitter grammar is missing operator spelling {spelling:?}")]
    MissingOperator { spelling: String },
    #[error("tree-sitter grammar has no pipe expression rule")]
    MissingPipeRule,
    #[error("tree-sitter grammar contract has no explicit CSG intersection precedence")]
    MissingCsgIntersection,
}

/// Strip full-line `//` comments from a tree-sitter grammar source before
/// contract matching.
///
/// A commented-out precedence line (e.g. `// prec.left(3, …)`) must not
/// satisfy the contract, and comment text must not provide operator
/// spellings. Only whole lines are stripped: inline `//` sequences inside
/// string/regex literals (e.g. the comment-token rules, which contain `/`
/// followed by `*` inside character classes) are never cut, which a naive
/// `/* … */` stripper would corrupt.
fn strip_js_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        if body.trim_start().starts_with("//") {
            out.push('\n');
        } else {
            out.push_str(line);
            if !line.ends_with('\n') {
                out.push('\n');
            }
        }
    }
    out
}

/// Validate the checked-in tree-sitter grammar against the Rust grammar rows.
///
/// This is deliberately source-level: consumers do not need Node.js or a
/// generated parser merely to detect editor grammar drift in CI or QA.
/// Matching runs on the comment-stripped source so commented-out rules do
/// not satisfy the contract. (Indentation-sensitive markers remain: a
/// reformatted grammar.js still needs the canonical 4-space rule layout.)
pub fn validate_tree_sitter_grammar(source: &str) -> Result<(), GrammarContractError> {
    let stripped = strip_js_comments(source);
    let source = stripped.as_str();
    if !source.contains("    pipe_expr:") || !source.contains("'|>'") {
        return Err(GrammarContractError::MissingPipeRule);
    }

    for spelling in OPERATOR_SPELLINGS {
        if !source.contains(&format!("'{spelling}'")) {
            return Err(GrammarContractError::MissingOperator {
                spelling: (*spelling).to_owned(),
            });
        }
    }

    let mut previous_rule_end = 0;
    for (index, (rule, precedence)) in TREE_SITTER_PRECEDENCE.iter().enumerate() {
        let start_marker = format!("    {rule}:");
        let start = if *rule == "constructor_expr" {
            source.find("\nconstructor_expr:").map(|offset| offset + 1)
        } else {
            source.find(&start_marker)
        }
        .ok_or_else(|| GrammarContractError::MissingRule {
            rule: (*rule).to_owned(),
        })?;
        if start < previous_rule_end {
            return Err(GrammarContractError::UnexpectedPrecedence {
                rule: (*rule).to_owned(),
                values: vec![*precedence],
            });
        }
        let next_rule = if *rule == "postfix_expr" {
            Some("primary_expr")
        } else if *rule == "constructor_expr" {
            Some("paren_expr")
        } else {
            TREE_SITTER_PRECEDENCE.get(index + 1).map(|(next, _)| *next)
        };
        let end = next_rule
            .and_then(|next| {
                let marker = if next == "primary_expr" {
                    "\n  primary_expr:".to_owned()
                } else {
                    format!("\n    {next}:")
                };
                source[start..].find(&marker)
            })
            .map(|offset| start + offset)
            .unwrap_or(source.len());
        let body = &source[start..end];
        let left = format!("prec.left({precedence},");
        let plain = format!("prec({precedence},");
        let right = format!("prec.right({precedence},");
        if !body.contains(&left) && !body.contains(&plain) && !body.contains(&right) {
            return Err(GrammarContractError::MissingPrecedence {
                rule: (*rule).to_owned(),
                precedence: *precedence,
            });
        }
        let mut values = Vec::new();
        for candidate in 0..=32 {
            if body.contains(&format!("prec.left({candidate},"))
                || body.contains(&format!("prec({candidate},"))
                || body.contains(&format!("prec.right({candidate},"))
            {
                values.push(candidate);
            }
        }
        if values.iter().any(|value| *value != *precedence) {
            return Err(GrammarContractError::UnexpectedPrecedence {
                rule: (*rule).to_owned(),
                values,
            });
        }
        previous_rule_end = end;
    }

    let csg_start =
        source
            .find("    csg_expr:")
            .ok_or_else(|| GrammarContractError::MissingRule {
                rule: "csg_expr".into(),
            })?;
    let csg_end = source[csg_start..]
        .find("\n    or_expr:")
        .map(|offset| csg_start + offset)
        .unwrap_or(source.len());
    let csg = &source[csg_start..csg_end];
    if !csg.contains("seq($.csg_expr, '&', $.or_expr)") {
        return Err(GrammarContractError::MissingCsgIntersection);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_contract_is_monotone_and_power_is_only_right_associative() {
        assert_eq!(PRECEDENCE_ROWS.len(), 11);
        assert!(PRECEDENCE_ROWS.windows(2).all(|rows| rows[0].1 < rows[1].1));
        assert_eq!(PRECEDENCE_ROWS.iter().filter(|row| row.2).count(), 1);
        assert_eq!(PRECEDENCE_ROWS[8].0, "power");
        assert!(!PRECEDENCE_ROWS.iter().any(|row| row.0 == "range"));
    }

    #[test]
    fn longest_match_contract_has_no_prefix_before_its_longer_operator() {
        for (shorter_index, shorter) in OPERATOR_SPELLINGS.iter().enumerate() {
            for (longer_index, longer) in OPERATOR_SPELLINGS.iter().enumerate() {
                if longer.len() > shorter.len() && longer.starts_with(shorter) {
                    assert!(
                        longer_index < shorter_index,
                        "{shorter} appears before its longer spelling {longer}"
                    );
                }
            }
        }
    }

    #[test]
    fn tree_sitter_grammar_matches_precedence_and_csg_contract() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tree-sitter-meshlang/grammar.js"
        ));
        validate_tree_sitter_grammar(source).expect("tree-sitter grammar must stay aligned");
    }

    #[test]
    fn tree_sitter_grammar_rejects_csg_precedence_drift() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tree-sitter-meshlang/grammar.js"
        ));
        let drifted = source.replace(
            "prec.left(3, seq($.csg_expr, '&', $.or_expr))",
            "prec.left(4, seq($.csg_expr, '&', $.or_expr))",
        );
        let error = validate_tree_sitter_grammar(&drifted).unwrap_err();
        assert!(matches!(
            error,
            GrammarContractError::UnexpectedPrecedence { .. }
        ));
    }

    #[test]
    fn tree_sitter_grammar_rejects_missing_operator() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tree-sitter-meshlang/grammar.js"
        ));
        let drifted = source.replace("'|>'", "'pipe-arrow'");
        assert!(matches!(
            validate_tree_sitter_grammar(&drifted),
            Err(GrammarContractError::MissingOperator { .. })
                | Err(GrammarContractError::MissingPipeRule)
        ));
    }

    #[test]
    fn grammar_contract_json_is_versioned_and_validated() {
        let json = crate::grammar_contract_json().expect("grammar contract JSON");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid grammar JSON");
        assert_eq!(
            value["schema_version"],
            serde_json::json!(crate::GRAMMAR_CONTRACT_SCHEMA_VERSION)
        );
        assert_eq!(value["precedence_rows"].as_array().unwrap().len(), 11);
        assert_eq!(
            value["operator_spellings"].as_array().unwrap().len(),
            OPERATOR_SPELLINGS.len()
        );
        assert!(OPERATOR_SPELLINGS.contains(&"=="));
        assert!(OPERATOR_SPELLINGS.contains(&"..="));
    }

    #[test]
    fn tree_sitter_grammar_ignores_commented_out_rules() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tree-sitter-meshlang/grammar.js"
        ));
        // A commented-out precedence line must not satisfy the contract.
        // (The comparison rule carries a single prec line whose operator
        // spellings live in `comparison_op`, so commenting it removes the
        // rule's precedence without tripping the spelling check first.)
        let commented = source.replace(
            "    prec.left(6, seq($.additive_expr, $.comparison_op, $.additive_expr)),",
            "    // prec.left(6, seq($.additive_expr, $.comparison_op, $.additive_expr)),",
        );
        assert!(matches!(
            validate_tree_sitter_grammar(&commented),
            Err(GrammarContractError::MissingPrecedence { .. })
        ));
        // The real grammar still validates after comment stripping.
        validate_tree_sitter_grammar(source).expect("comments must not break validation");
    }
}
