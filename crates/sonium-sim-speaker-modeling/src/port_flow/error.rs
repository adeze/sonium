//! Error type for the resolved nonlinear port-flow module.

/// Errors raised by the port-flow geometry, meshing, and (later) N-S solve stages.
#[derive(Debug, thiserror::Error)]
pub enum PortFlowError {
    #[error("port mesh error: {0}")]
    Mesh(String),
    #[error("N-S solver failed at step {step}, t = {time_s} s")]
    SolverDiverged { step: usize, time_s: f64 },
    #[error("invalid port-flow configuration: {0}")]
    InvalidConfig(String),
    #[error("round-port linear reference failed: {0}")]
    LinearReference(String),
}
