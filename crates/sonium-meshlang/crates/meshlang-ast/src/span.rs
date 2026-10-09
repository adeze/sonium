//! Source location tracking for error reporting.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Byte offset range in source text.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    /// Optional file ID for multi-file compilation.
    pub file_id: u32,
}

impl fmt::Display for Span {
    /// User-facing location (`file {id} bytes {start}..{end}`); prefer this
    /// over `{:?}` in diagnostics so errors never leak the `Span { .. }`
    /// debug struct. Byte offsets (not line/col) — resolve through
    /// `LineIndex` where line:col is needed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "file {} bytes {}..{}",
            self.file_id, self.start, self.end
        )
    }
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            file_id: 0,
        }
    }

    pub fn with_file(start: usize, end: usize, file_id: u32) -> Self {
        Self {
            start,
            end,
            file_id,
        }
    }

    /// Merge two same-file spans, panicking on cross-file input.
    ///
    /// Only use this when both spans are known to come from one file (as
    /// the single-file parser does). Cross-file merges are a normal
    /// multi-file-compilation event: prefer [`Self::try_merge`] there so a
    /// recoverable diagnostic does not become a process panic.
    pub fn merge(self, other: Span) -> Span {
        self.try_merge(other)
            .expect("cannot merge spans from different source files")
    }

    pub fn try_merge(self, other: Span) -> Result<Span, SpanMergeError> {
        if self.file_id != other.file_id {
            return Err(SpanMergeError {
                left_file: self.file_id,
                right_file: other.file_id,
            });
        }
        Ok(Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            file_id: self.file_id,
        })
    }

    /// Byte length. Saturates to 0 for inverted spans (`start > end`)
    /// instead of panicking in debug or wrapping in release; see
    /// [`Self::is_valid`].
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Whether the span is well-formed (`start <= end`). Validators
    /// elsewhere already observe inverted spans reaching this crate, so
    /// accessors must not assume validity.
    pub fn is_valid(&self) -> bool {
        self.start <= self.end
    }

    /// Dummy span for synthesized nodes.
    pub const DUMMY: Span = Span {
        start: 0,
        end: 0,
        file_id: u32::MAX,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanMergeError {
    pub left_file: u32,
    pub right_file: u32,
}

impl fmt::Display for SpanMergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot merge spans from source files {} and {}",
            self.left_file, self.right_file
        )
    }
}

impl std::error::Error for SpanMergeError {}

impl Default for Span {
    fn default() -> Self {
        Self::DUMMY
    }
}

/// Resolved line/column for display.
#[derive(Debug, Clone, Copy)]
pub struct LineCol {
    pub line: u32,   // 1-based
    pub column: u32, // 1-based
}

/// Maps byte offsets to line/column in a source file.
pub struct LineIndex {
    line_starts: Vec<usize>,
    scalar_starts: Vec<usize>,
}

impl LineIndex {
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        let mut scalar_starts = Vec::new();
        for (i, c) in source.char_indices() {
            scalar_starts.push(i);
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            scalar_starts,
        }
    }

    pub fn line_col(&self, offset: usize) -> LineCol {
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let line_start = self.line_starts[line];
        let scalars_before_offset = self
            .scalar_starts
            .partition_point(|&scalar_start| scalar_start < offset);
        let scalars_before_line = self
            .scalar_starts
            .partition_point(|&scalar_start| scalar_start < line_start);
        let column = scalars_before_offset.saturating_sub(scalars_before_line);
        LineCol {
            line: line as u32 + 1,
            column: column as u32 + 1,
        }
    }

    /// Get the source text for a given line (0-indexed).
    pub fn line_text<'a>(&self, source: &'a str, line: usize) -> &'a str {
        let start = self.line_starts[line];
        let end = self
            .line_starts
            .get(line + 1)
            .copied()
            .unwrap_or(source.len());
        let text = &source[start..end];
        let text = text.strip_suffix('\n').unwrap_or(text);
        text.strip_suffix('\r').unwrap_or(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_merge_is_commutative_and_enclosing() {
        let first = Span::with_file(4, 8, 2);
        let second = Span::with_file(1, 6, 2);
        assert_eq!(
            first.try_merge(second).unwrap(),
            second.try_merge(first).unwrap()
        );
        assert_eq!(first.merge(second), Span::with_file(1, 8, 2));
    }

    #[test]
    fn inverted_spans_saturate_instead_of_panicking() {
        let bad = Span::new(8, 4);
        assert!(!bad.is_valid());
        assert_eq!(bad.len(), 0);
        assert!(bad.is_empty());
        let good = Span::new(4, 8);
        assert!(good.is_valid());
        assert_eq!(good.len(), 4);
        assert!(!good.is_empty());
    }

    #[test]
    fn cross_source_merge_is_an_explicit_error() {
        let error = Span::with_file(0, 1, 1)
            .try_merge(Span::with_file(0, 1, 2))
            .unwrap_err();
        assert_eq!(error.left_file, 1);
        assert_eq!(error.right_file, 2);
    }

    #[test]
    fn unicode_columns_count_scalars_not_utf8_bytes() {
        let source = "é🚀x\n";
        let index = LineIndex::new(source);
        assert_eq!(index.line_col(0).column, 1);
        assert_eq!(index.line_col(2).column, 2);
        assert_eq!(index.line_col(6).column, 3);
    }

    #[test]
    fn line_text_strips_both_lf_and_crlf_terminators() {
        let source = "first\r\nsecond\n";
        let index = LineIndex::new(source);
        assert_eq!(index.line_text(source, 0), "first");
        assert_eq!(index.line_text(source, 1), "second");
    }
}
