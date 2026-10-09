//! LSP-compatible diagnostic output.
//!
//! Emits diagnostics in formats consumable by editors:
//! - JSON (for LSP servers and tooling)
//! - Human-readable (for terminal output, rustc-style)
//! - SARIF (for CI/CD integration)

use crate::type_check::{DiagLevel, Diagnostic};
use meshlang_ast::span::{LineIndex, Span};
use serde::{Deserialize, Serialize};

/// LSP Diagnostic severity levels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LspSeverity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

impl From<DiagLevel> for LspSeverity {
    fn from(level: DiagLevel) -> Self {
        match level {
            DiagLevel::Error => LspSeverity::Error,
            DiagLevel::Warning => LspSeverity::Warning,
            DiagLevel::Info => LspSeverity::Information,
        }
    }
}

/// LSP-compatible position (0-indexed line and character).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspPosition {
    pub line: u32,
    pub character: u32,
}

/// LSP-compatible range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspRange {
    pub start: LspPosition,
    pub end: LspPosition,
}

/// LSP-compatible diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspDiagnostic {
    pub range: LspRange,
    pub severity: u32,
    pub source: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(rename = "relatedInformation", skip_serializing_if = "Option::is_none")]
    pub related_information: Option<Vec<LspRelatedInfo>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspRelatedInfo {
    pub location: LspLocation,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspLocation {
    pub uri: String,
    pub range: LspRange,
}

/// Convert internal diagnostics to LSP format.
pub fn to_lsp_diagnostics(
    diagnostics: &[Diagnostic],
    source: &str,
    file_uri: &str,
) -> Vec<LspDiagnostic> {
    let line_index = LineIndex::new(source);

    diagnostics
        .iter()
        .map(|d| {
            let start_loc = if d.span != Span::DUMMY {
                line_index.line_col(d.span.start)
            } else {
                meshlang_ast::span::LineCol { line: 1, column: 1 }
            };
            let end_loc = if d.span != Span::DUMMY {
                line_index.line_col(d.span.end)
            } else {
                meshlang_ast::span::LineCol { line: 1, column: 1 }
            };

            // `LineCol` is 1-based by construction and the DUMMY span is
            // guarded above, so these never underflow; `saturating_sub`
            // hardens against any future 0-based path for free.
            let start = LspPosition {
                line: start_loc.line.saturating_sub(1),
                character: start_loc.column.saturating_sub(1),
            };
            let end = LspPosition {
                line: end_loc.line.saturating_sub(1),
                character: end_loc.column.saturating_sub(1),
            };
            let related = d.hint.as_ref().map(|hint| {
                vec![LspRelatedInfo {
                    location: LspLocation {
                        uri: file_uri.to_string(),
                        range: LspRange {
                            start: start.clone(),
                            end: end.clone(),
                        },
                    },
                    message: hint.clone(),
                }]
            });

            LspDiagnostic {
                range: LspRange { start, end },
                severity: LspSeverity::from(d.level) as u32,
                source: "meshlang".to_string(),
                message: d.message.clone(),
                code: None,
                related_information: related,
            }
        })
        .collect()
}

/// Whether terminal formatting should emit ANSI escapes.
///
/// Honors the [`NO_COLOR`](https://no-color.org) convention: any `NO_COLOR`
/// environment value (even empty) disables color. Piped/CI callers that
/// cannot rely on the environment should use [`format_terminal_with_color`].
fn terminal_color_enabled() -> bool {
    std::env::var("NO_COLOR").is_err()
}

/// Format diagnostics for terminal output (rustc-style).
///
/// Color follows [`terminal_color_enabled`](self::terminal_color_enabled);
/// use [`format_terminal_with_color`] for explicit control.
pub fn format_terminal(diagnostics: &[Diagnostic], source: &str, filename: &str) -> String {
    format_terminal_with_color(diagnostics, source, filename, terminal_color_enabled())
}

/// [`format_terminal`] with an explicit color switch for piped/CI contexts.
pub fn format_terminal_with_color(
    diagnostics: &[Diagnostic],
    source: &str,
    filename: &str,
    color: bool,
) -> String {
    let line_index = LineIndex::new(source);
    let mut out = String::new();
    let paint = |code: &str, text: &str| -> String {
        if color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    };

    for d in diagnostics {
        let prefix = match d.level {
            DiagLevel::Error => paint("1;31", "error"),
            DiagLevel::Warning => paint("1;33", "warning"),
            DiagLevel::Info => paint("1;34", "info"),
        };

        if d.span != Span::DUMMY {
            let loc = line_index.line_col(d.span.start);
            let line_text = line_index.line_text(source, loc.line.saturating_sub(1) as usize);

            out.push_str(&format!("{}: {}\n", prefix, d.message));
            out.push_str(&format!(
                "  {} {}:{}:{}\n",
                paint("1;34", "-->"),
                filename,
                loc.line,
                loc.column
            ));
            out.push_str(&format!("   {}\n", paint("1;34", "|")));
            out.push_str(&format!(
                "{} {}\n",
                paint("1;34", &format!("{:>3} |", loc.line)),
                line_text
            ));

            // Underline width is a *character* count over the span's byte
            // range (mirroring the parser's `display_with_source`); byte
            // lengths overshoot on multi-byte spans.
            let underline_len = source
                .get(d.span.start..d.span.end)
                .map_or(1, |fragment| fragment.chars().count().max(1));
            let carets = "^".repeat(underline_len);
            let painted_carets = match d.level {
                DiagLevel::Error => paint("1;31", &carets),
                DiagLevel::Warning => paint("1;33", &carets),
                DiagLevel::Info => paint("1;34", &carets),
            };
            out.push_str(&format!(
                "   {} {}{}",
                paint("1;34", "|"),
                " ".repeat(loc.column.saturating_sub(1) as usize),
                painted_carets
            ));
            out.push('\n');
        } else {
            out.push_str(&format!("{}: {}\n", prefix, d.message));
        }

        if let Some(ref hint) = d.hint {
            out.push_str(&format!("   {} {}\n", paint("1;34", "= hint:"), hint));
        }

        out.push('\n');
    }

    out
}

/// Fallible form of [`to_json`]: serialization of the LSP structs cannot
/// fail in practice, but tool callers must be able to distinguish "no
/// diagnostics" (`Ok("[]")`) from a serialization failure (`Err`).
pub fn try_to_json(
    diagnostics: &[Diagnostic],
    source: &str,
    file_uri: &str,
) -> Result<String, serde_json::Error> {
    let lsp_diags = to_lsp_diagnostics(diagnostics, source, file_uri);
    serde_json::to_string_pretty(&lsp_diags)
}

/// Emit diagnostics as JSON (for tool integration).
///
/// Falls back to `"[]"` (not `""`) if serialization ever fails, so the
/// result stays valid JSON. Use [`try_to_json`] when the failure itself
/// must be observable.
pub fn to_json(diagnostics: &[Diagnostic], source: &str, file_uri: &str) -> String {
    try_to_json(diagnostics, source, file_uri).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_at(message: &str, start: usize, end: usize) -> Diagnostic {
        Diagnostic {
            level: DiagLevel::Error,
            message: message.to_string(),
            span: Span::with_file(start, end, 0),
            hint: None,
        }
    }

    #[test]
    fn terminal_underline_counts_characters_not_bytes() {
        // "é" is 2 bytes / 1 char: the caret run must be 1 wide, and the
        // padding must align to the 1-based character column.
        let source = "param é = 1\n";
        let start = source.find('é').unwrap();
        let end = start + 'é'.len_utf8();
        let out =
            format_terminal_with_color(&[error_at("bad", start, end)], source, "u.mlg", false);
        let carets = out.lines().find(|line| line.contains('^')).unwrap();
        assert_eq!(carets.chars().filter(|&c| c == '^').count(), 1);
        assert!(!out.contains('\x1b'));
    }

    #[test]
    fn terminal_no_color_mode_emits_no_escapes() {
        let source = "param x = 1\n";
        let out = format_terminal_with_color(&[error_at("bad", 6, 7)], source, "c.mlg", false);
        assert!(!out.contains('\x1b'));
        assert!(out.contains("error: bad"));
    }

    #[test]
    fn json_round_trip_distinguishes_empty_from_failure() {
        let json = try_to_json(&[], "param x = 1\n", "file://c.mlg").unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value, serde_json::json!([]));
        let legacy = to_json(&[], "param x = 1\n", "file://c.mlg");
        assert_eq!(legacy, json);
    }

    #[test]
    fn lsp_conversion_saturates_instead_of_underflowing() {
        // DUMMY spans map to 1:1 → 0:0 without underflow.
        let dummy = Diagnostic {
            level: DiagLevel::Warning,
            message: "w".to_string(),
            span: Span::DUMMY,
            hint: Some("h".to_string()),
        };
        let lsp = to_lsp_diagnostics(&[dummy], "", "file://d.mlg");
        assert_eq!(lsp[0].range.start.line, 0);
        assert_eq!(lsp[0].range.start.character, 0);
    }
}
