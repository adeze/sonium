use thiserror::Error;

/// Errors that can occur during autodiff module operations.
#[derive(Debug, Error)]
pub enum AutodiffError {
    /// A generic error with a descriptive message.
    #[error("{0}")]
    Message(String),
}
