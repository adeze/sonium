use meshlang_ast::span::Span;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("parse error: {0}")]
    Parse(String),

    #[error("undefined reference: '{name}'")]
    UndefinedReference { name: String },

    #[error("duplicate definition: '{name}' at {span}")]
    DuplicateDefinition { name: String, span: Span },

    #[error("type error: {message}")]
    TypeError { message: String },

    #[error("unit mismatch: cannot {op} {lhs} and {rhs}")]
    UnitMismatch {
        op: String,
        lhs: String,
        rhs: String,
    },

    #[error("unsupported feature: {0}")]
    Unsupported(String),

    #[error("internal compiler error: {0}")]
    Internal(String),
}
