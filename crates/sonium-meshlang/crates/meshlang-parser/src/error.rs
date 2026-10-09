//! Parser error types with source location.

use meshlang_ast::span::{LineIndex, Span};
use serde::Serialize;
use thiserror::Error;

/// Wire/schema version for [`DiagnosticCode`] and serialized [`ParseError`].
pub const DIAGNOSTIC_SCHEMA_VERSION: u32 = 2;

/// Stable, machine-readable parser diagnostic categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticCode {
    Lexical,
    UnexpectedToken,
    InvalidNumber,
    InvalidUnit,
    UnsupportedVersion,
    NestingLimit,
    TooManyDiagnostics,
    SourceTooLarge,
}

#[derive(Debug, Clone, Error, Serialize)]
#[error("{message}")]
pub struct ParseError {
    pub code: DiagnosticCode,
    pub message: String,
    pub span: Span,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_name: Option<String>,
}

impl ParseError {
    pub fn new(code: DiagnosticCode, message: impl Into<String>, span: Span) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            source_name: None,
        }
    }

    pub fn with_source_name(mut self, source_name: impl Into<String>) -> Self {
        self.source_name = Some(source_name.into());
        self
    }

    /// Format error with source context, Rust-compiler style.
    pub fn display_with_source(&self, source: &str, filename: &str) -> String {
        let idx = LineIndex::new(source);
        let loc = idx.line_col(self.span.start);
        let line_text = idx.line_text(source, loc.line as usize - 1);

        let mut out = String::new();
        out.push_str(&format!(
            "error: {}\n  --> {}:{}:{}\n",
            self.message, filename, loc.line, loc.column
        ));
        out.push_str("   |\n");
        out.push_str(&format!("{:>3} | {}\n", loc.line, line_text));
        let underline_width = source
            .get(self.span.start..self.span.end)
            .map_or(1, |fragment| fragment.chars().count().max(1));
        out.push_str(&format!(
            "   | {}{}",
            " ".repeat((loc.column as usize).saturating_sub(1)),
            "^".repeat(underline_width)
        ));
        out.push('\n');
        out
    }
}
