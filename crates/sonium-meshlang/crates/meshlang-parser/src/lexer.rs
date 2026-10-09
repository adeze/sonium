//! Lexer: source text → token stream.

use std::collections::HashMap;
use std::sync::Arc;

use meshlang_ast::provenance::{Comment, CommentKind};
use meshlang_ast::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // Literals
    Integer,
    Float,
    StringLit,
    True,
    False,

    // Identifiers & type identifiers
    Ident,     // lowercase start: foo_bar
    TypeIdent, // uppercase start: Box, Cylinder

    // Keywords
    Param,
    Const,
    Import,
    From,
    As,
    Fn,
    Pub,
    Geometry,
    Mesh,
    Physics,
    Export,
    Struct,
    Let,
    For,
    In,
    If,
    Else,
    Group,
    Refine,
    Near,
    On,
    BoundaryLayer,
    Default,
    Transfinite,
    Embed,
    Periodic,
    Compound,
    Version,
    Design,
    Uncertainty,
    Objective,
    Constraint,
    DesignRegion,
    PhysicalRegion,
    Boundary,
    Interface,
    CoordinateSystem,
    MaterialRegion,
    Field,
    Metric,

    // Units (recognized as keywords for disambiguation)
    // Actual unit parsing happens in the parser using ident text

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    StarStar,  // **
    Pipe,      // |
    Ampersand, // &
    PipeArrow, // |>
    Eq,        // ==
    Ne,        // !=
    Lt,        // <
    Gt,        // >
    Le,        // <=
    Ge,        // >=
    AmpAmp,    // &&
    PipePipe,  // ||
    Bang,      // !
    Assign,    // =
    Arrow,     // ->
    DotDot,    // ..
    DotDotEq,  // ..=

    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Dot,
    Semicolon,

    // Special
    Eof,
    Error,
}

/// Stable classification for lexical failures.
///
/// Keeping the category separate from the rendered message lets editor and
/// WASM clients react to diagnostics without parsing English text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexErrorKind {
    UnexpectedCharacter,
    UnterminatedString,
    InvalidEscape,
    NewlineInString,
    UnterminatedBlockComment,
    MalformedNumber,
}

impl LexErrorKind {
    pub fn message(self) -> &'static str {
        match self {
            Self::UnexpectedCharacter => "unexpected character",
            Self::UnterminatedString => "unterminated string literal",
            Self::InvalidEscape => "invalid string escape",
            Self::NewlineInString => "newline in string literal",
            Self::UnterminatedBlockComment => "unterminated block comment",
            Self::MalformedNumber => "malformed numeric literal",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// Interned source spelling. Repeated keywords, punctuation, identifiers,
    /// and literals share one allocation within a lexing session.
    pub text: Arc<str>,
    pub error: Option<LexErrorKind>,
}

pub struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    pos: usize,
    file_id: u32,
    /// Byte offset of `source` within the caller's original file.
    ///
    /// Subtree parsing lexes a borrowed source slice to avoid copying the
    /// complete file. Keeping the offset here makes every token/comment span
    /// absolute while all text slicing remains local to the borrowed slice.
    source_offset: usize,
    interned: HashMap<&'a str, Arc<str>>,
    comments: Vec<Comment>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str, file_id: u32) -> Self {
        Self::with_offset(source, file_id, 0)
    }

    pub fn with_offset(source: &'a str, file_id: u32, source_offset: usize) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            pos: 0,
            file_id,
            source_offset,
            interned: HashMap::new(),
            comments: Vec::new(),
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        tokens
    }

    pub fn take_comments(&mut self) -> Vec<Comment> {
        std::mem::take(&mut self.comments)
    }

    fn next_token(&mut self) -> Token {
        if let Some(error) = self.skip_whitespace_and_comments() {
            return error;
        }

        if self.pos >= self.bytes.len() {
            return Token {
                kind: TokenKind::Eof,
                span: Span::with_file(
                    self.source_offset + self.pos,
                    self.source_offset + self.pos,
                    self.file_id,
                ),
                text: Arc::from(""),
                error: None,
            };
        }

        let start = self.pos;
        let byte = self.bytes[self.pos];
        let ch = self.source[self.pos..].chars().next().unwrap_or('\u{fffd}');

        // String literals
        if byte == b'"' {
            return self.lex_string(start);
        }

        // Numbers
        if byte.is_ascii_digit()
            || (byte == b'.' && self.peek(1).is_some_and(|b| b.is_ascii_digit()))
        {
            return self.lex_number(start);
        }

        // Identifiers and keywords
        if ch.is_alphabetic() || byte == b'_' {
            return self.lex_ident(start);
        }

        // Two-character operators
        if let Some(kind) = self.try_two_char(start) {
            return kind;
        }

        // Single-character operators and punctuation
        let kind = match byte {
            b'+' => TokenKind::Plus,
            b'-' => TokenKind::Minus,
            b'*' => TokenKind::Star,
            b'/' => TokenKind::Slash,
            b'%' => TokenKind::Percent,
            b'^' => TokenKind::Caret,
            b'|' => TokenKind::Pipe,
            b'&' => TokenKind::Ampersand,
            b'!' => TokenKind::Bang,
            b'<' => TokenKind::Lt,
            b'>' => TokenKind::Gt,
            b'=' => TokenKind::Assign,
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b',' => TokenKind::Comma,
            b':' => TokenKind::Colon,
            b'.' => TokenKind::Dot,
            b';' => TokenKind::Semicolon,
            _ => {
                self.pos += ch.len_utf8();
                return self.make_error(LexErrorKind::UnexpectedCharacter, start);
            }
        };

        self.pos += 1;

        self.make_token(kind, start)
    }

    fn lex_string(&mut self, start: usize) -> Token {
        self.pos += 1; // skip opening "
        while self.pos < self.bytes.len() {
            let ch = self.source[self.pos..].chars().next().unwrap_or('\u{fffd}');
            match ch {
                '"' => {
                    self.pos += 1;
                    return self.make_token(TokenKind::StringLit, start);
                }
                '\n' | '\r' => {
                    return self.make_error(LexErrorKind::NewlineInString, start);
                }
                '\\' => {
                    self.pos += 1;
                    let Some(escaped) = self.source[self.pos..].chars().next() else {
                        return self.make_error(LexErrorKind::UnterminatedString, start);
                    };
                    self.pos += escaped.len_utf8();
                    if !matches!(escaped, '"' | '\\' | 'n' | 'r' | 't' | '0') {
                        return self.make_error(LexErrorKind::InvalidEscape, start);
                    }
                }
                _ => self.pos += ch.len_utf8(),
            }
        }
        self.make_error(LexErrorKind::UnterminatedString, start)
    }

    fn lex_number(&mut self, start: usize) -> Token {
        let mut is_float = false;

        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }

        // Decimal part
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' {
            // Check it's not `..` (range)
            if self.peek(1) != Some(b'.') {
                is_float = true;
                self.pos += 1;
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            }
        }

        // Exponent
        if self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E')
        {
            is_float = true;
            self.pos += 1;
            if self.pos < self.bytes.len()
                && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-')
            {
                self.pos += 1;
            }
            let exponent_start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos == exponent_start {
                return self.make_error(LexErrorKind::MalformedNumber, start);
            }
        }

        let kind = if is_float {
            TokenKind::Float
        } else {
            TokenKind::Integer
        };
        self.make_token(kind, start)
    }

    fn lex_ident(&mut self, start: usize) -> Token {
        while self.pos < self.bytes.len() {
            let ch = self.source[self.pos..].chars().next().unwrap_or('\u{fffd}');
            if ch.is_alphanumeric() || ch == '_' {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }

        let text = &self.source[start..self.pos];
        let kind = match text {
            "param" => TokenKind::Param,
            "const" => TokenKind::Const,
            "import" => TokenKind::Import,
            "from" => TokenKind::From,
            "as" => TokenKind::As,
            "fn" => TokenKind::Fn,
            "pub" => TokenKind::Pub,
            "geometry" => TokenKind::Geometry,
            "mesh" => TokenKind::Mesh,
            "physics" => TokenKind::Physics,
            "export" => TokenKind::Export,
            "struct" => TokenKind::Struct,
            "let" => TokenKind::Let,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "group" => TokenKind::Group,
            "refine" => TokenKind::Refine,
            "near" => TokenKind::Near,
            "on" => TokenKind::On,
            "boundary_layer" => TokenKind::BoundaryLayer,
            "default" => TokenKind::Default,
            "transfinite" => TokenKind::Transfinite,
            "embed" => TokenKind::Embed,
            "periodic" => TokenKind::Periodic,
            "compound" => TokenKind::Compound,
            "version" => TokenKind::Version,
            "design" => TokenKind::Design,
            "uncertainty" => TokenKind::Uncertainty,
            "objective" => TokenKind::Objective,
            "constraint" => TokenKind::Constraint,
            "design_region" => TokenKind::DesignRegion,
            "physical_region" => TokenKind::PhysicalRegion,
            "boundary" => TokenKind::Boundary,
            "interface" => TokenKind::Interface,
            "coordinate_system" => TokenKind::CoordinateSystem,
            "material_region" => TokenKind::MaterialRegion,
            "field" => TokenKind::Field,
            "metric" => TokenKind::Metric,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => {
                // Uppercase start → type identifier
                if text.chars().next().is_some_and(char::is_uppercase) {
                    TokenKind::TypeIdent
                } else {
                    TokenKind::Ident
                }
            }
        };

        self.make_token(kind, start)
    }

    fn try_two_char(&mut self, start: usize) -> Option<Token> {
        let c1 = self.bytes[self.pos];
        let c2 = self.peek(1)?;

        let kind = match (c1, c2) {
            (b'*', b'*') => TokenKind::StarStar,
            (b'|', b'>') => TokenKind::PipeArrow,
            (b'=', b'=') => TokenKind::Eq,
            (b'!', b'=') => TokenKind::Ne,
            (b'<', b'=') => TokenKind::Le,
            (b'>', b'=') => TokenKind::Ge,
            (b'&', b'&') => TokenKind::AmpAmp,
            (b'|', b'|') => TokenKind::PipePipe,
            (b'-', b'>') => TokenKind::Arrow,
            (b'.', b'.') => {
                // Could be .. or ..=
                if self.peek(2) == Some(b'=') {
                    self.pos += 3;
                    return Some(self.make_token(TokenKind::DotDotEq, start));
                }
                TokenKind::DotDot
            }
            _ => return None,
        };

        self.pos += 2;
        Some(self.make_token(kind, start))
    }

    fn skip_whitespace_and_comments(&mut self) -> Option<Token> {
        loop {
            // Whitespace
            while self.pos < self.bytes.len() {
                let ch = self.source[self.pos..].chars().next().unwrap_or('\u{fffd}');
                if ch.is_whitespace() {
                    self.pos += ch.len_utf8();
                } else {
                    break;
                }
            }

            if self.pos >= self.bytes.len() {
                return None;
            }

            // Line and documentation comments are lossless trivia everywhere,
            // including inside blocks and expression lists.
            if self.bytes[self.pos] == b'/' && self.peek(1) == Some(b'/') {
                let start = self.pos;
                let kind = if self.peek(2) == Some(b'/') {
                    CommentKind::DocLine
                } else {
                    CommentKind::Line
                };
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                self.record_comment(kind, start);
                continue;
            }

            // Block comment
            if self.bytes[self.pos] == b'/' && self.peek(1) == Some(b'*') {
                let start = self.pos;
                let kind = if self.peek(2) == Some(b'*') {
                    CommentKind::DocBlock
                } else {
                    CommentKind::Block
                };
                self.pos += 2;
                let mut depth = 1;
                while self.pos < self.bytes.len() && depth > 0 {
                    if self.bytes[self.pos] == b'/' && self.peek(1) == Some(b'*') {
                        depth += 1;
                        self.pos += 2;
                    } else if self.bytes[self.pos] == b'*' && self.peek(1) == Some(b'/') {
                        depth -= 1;
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
                self.record_comment(kind, start);
                if depth != 0 {
                    return Some(self.make_error(LexErrorKind::UnterminatedBlockComment, start));
                }
                continue;
            }

            return None;
        }
    }

    fn peek(&self, offset: usize) -> Option<u8> {
        self.pos
            .checked_add(offset)
            .and_then(|position| self.bytes.get(position))
            .copied()
    }

    fn make_token(&mut self, kind: TokenKind, start: usize) -> Token {
        let text = self.source.get(start..self.pos).unwrap_or("");
        let interned = self
            .interned
            .entry(text)
            .or_insert_with(|| Arc::from(text))
            .clone();
        Token {
            kind,
            span: Span::with_file(
                self.source_offset + start,
                self.source_offset + self.pos,
                self.file_id,
            ),
            text: interned,
            error: None,
        }
    }

    fn make_error(&mut self, error: LexErrorKind, start: usize) -> Token {
        let mut token = self.make_token(TokenKind::Error, start);
        token.error = Some(error);
        token
    }

    fn record_comment(&mut self, kind: CommentKind, start: usize) {
        let text = self.source.get(start..self.pos).unwrap_or("");
        self.comments.push(Comment::new(
            kind,
            text,
            Span::with_file(
                self.source_offset + start,
                self.source_offset + self.pos,
                self.file_id,
            ),
        ));
    }

    #[cfg(test)]
    fn interned_spellings(&self) -> usize {
        self.interned.len()
    }
}

impl Token {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn lexical_error(&self) -> Option<LexErrorKind> {
        self.error
    }
}

#[cfg(test)]
impl Lexer<'_> {
    fn token_text_ptr(token: &Token) -> *const u8 {
        token.text.as_ptr()
    }
}

#[cfg(test)]
mod roadmap_tests {
    use super::*;

    #[test]
    fn unicode_never_splits_a_utf8_boundary() {
        let source = "param räume = 1\n🚀";
        let tokens = Lexer::new(source, 7).tokenize();
        for token in &tokens {
            assert!(source.is_char_boundary(token.span.start));
            assert!(source.is_char_boundary(token.span.end));
        }
        assert_eq!(tokens[1].text(), "räume");
        assert_eq!(
            tokens[4].lexical_error(),
            Some(LexErrorKind::UnexpectedCharacter)
        );
    }

    #[test]
    fn malformed_lexemes_are_typed_and_bounded() {
        for (source, expected) in [
            ("1e+", LexErrorKind::MalformedNumber),
            ("\"unterminated", LexErrorKind::UnterminatedString),
            ("\"bad\\q\"", LexErrorKind::InvalidEscape),
            ("/* unterminated", LexErrorKind::UnterminatedBlockComment),
        ] {
            let tokens = Lexer::new(source, 0).tokenize();
            assert_eq!(tokens[0].lexical_error(), Some(expected), "{source:?}");
            assert_eq!(tokens.last().map(|token| token.kind), Some(TokenKind::Eof));
        }
    }

    #[test]
    fn repeated_spellings_are_interned() {
        let mut lexer = Lexer::new("let same = same let same = same", 0);
        let tokens = lexer.tokenize();
        let same: Vec<_> = tokens
            .iter()
            .filter(|token| token.text() == "same")
            .collect();
        assert_eq!(same.len(), 4);
        assert!(
            same.windows(2)
                .all(|pair| Lexer::token_text_ptr(pair[0]) == Lexer::token_text_ptr(pair[1]))
        );
        assert!(lexer.interned_spellings() < tokens.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<(TokenKind, &str)> {
        let mut lexer = Lexer::new(src, 0);
        let tokens = lexer.tokenize();
        tokens
            .iter()
            .filter(|t| t.kind != TokenKind::Eof)
            .map(|t| (t.kind, &src[t.span.start..t.span.end]))
            .collect()
    }

    #[test]
    fn test_param_decl() {
        let tokens = lex("param freq = 1000.0 Hz");
        assert_eq!(tokens[0], (TokenKind::Param, "param"));
        assert_eq!(tokens[1], (TokenKind::Ident, "freq"));
        assert_eq!(tokens[2], (TokenKind::Assign, "="));
        assert_eq!(tokens[3], (TokenKind::Float, "1000.0"));
        assert_eq!(tokens[4], (TokenKind::TypeIdent, "Hz"));
    }

    #[test]
    fn test_pipe_arrow() {
        let tokens = lex("|> translate");
        assert_eq!(tokens[0], (TokenKind::PipeArrow, "|>"));
        assert_eq!(tokens[1], (TokenKind::Ident, "translate"));
    }

    #[test]
    fn test_csg_ops() {
        let tokens = lex("a - b | c & d");
        assert_eq!(tokens[1], (TokenKind::Minus, "-"));
        assert_eq!(tokens[3], (TokenKind::Pipe, "|"));
        assert_eq!(tokens[5], (TokenKind::Ampersand, "&"));
    }

    #[test]
    fn test_type_ident() {
        let tokens = lex("Box Cylinder foo bar");
        assert_eq!(tokens[0].0, TokenKind::TypeIdent);
        assert_eq!(tokens[1].0, TokenKind::TypeIdent);
        assert_eq!(tokens[2].0, TokenKind::Ident);
        assert_eq!(tokens[3].0, TokenKind::Ident);
    }

    #[test]
    fn test_compound_unit() {
        // m/s is actually m, /, s — unit parsing is in the parser layer
        let tokens = lex("343.0 m");
        assert_eq!(tokens[0], (TokenKind::Float, "343.0"));
        assert_eq!(tokens[1], (TokenKind::Ident, "m"));
    }

    #[test]
    fn test_leading_dot_float() {
        let tokens = lex(".5 .5e-2 .");
        assert_eq!(tokens[0], (TokenKind::Float, ".5"));
        assert_eq!(tokens[1], (TokenKind::Float, ".5e-2"));
        assert_eq!(tokens[2], (TokenKind::Dot, "."));
    }
}
