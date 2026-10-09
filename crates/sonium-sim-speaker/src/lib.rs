//! Loudspeaker simulation and optimization.
//!
//! Parametric model of a sealed loudspeaker cabinet with woofer, tweeter,
//! waveguide, and internal separator. Generates surface meshes, runs BEM
//! acoustic simulation, produces CEA-2034 measurements, and optimises
//! cabinet parameters for flat response and smooth radiation.

pub mod cea2034_output;
pub mod coupling;
pub mod directivity_export;
pub mod directivity_reference;
pub use directivity_reference::{
    DIRECTIVITY_REFERENCE_SCHEMA_VERSION, DirectivityCorrelationError,
    DirectivityCorrelationReport, DirectivityReference, DirectivityReferencePoint,
    DirectivityReferenceTolerances, correlate_directivity_reference,
};
pub mod distortion;
pub mod measurement_reference;
pub mod mesh_gen;
pub mod mlg_loader;
pub mod model;
pub mod product_metrics;
// Optimization uses the CEA-2034 PIR / preference-score helpers from
// sonium_roomeq::cea2034 via cea2034_output — none of which need the
// `cli` feature (no clap coupling), so the module is always available.
pub mod optimization;
pub mod power_compression;
pub mod simulation;
pub mod spline;

pub use simulation::{
    ArtifactFloatFormat, ArtifactOrdering, ExecutionProfile, GpuCacheReport, GpuCapabilityCell,
    GpuCapabilityMatrix, GpuCapabilityState, GpuExecutionReport, GpuFailureMode, GpuMemoryReport,
    GpuResidencyIdentity, GpuSpeakerConfigurationCapabilities, GpuTimingReport, GpuTransferReport,
    GpuValidationPolicy, GpuValidationReport, PrecisionPolicy, ResidualGateReport,
    ResidualGateStatus, SimulationArtifactMetadata, SimulationBackend, SimulationControl,
    SimulationError, SimulationProgress, SimulationProvenance, SimulationResult, SolverConfig,
    SpeakerPreviewArtifact, WASM_PREVIEW_MAX_EVALUATION_POINTS, WASM_PREVIEW_MAX_FREQUENCIES,
};

#[cfg(feature = "cuda")]
pub use simulation::{
    GpuAdjointReport, run_simulation_cuda_adjoint, run_simulation_cuda_adjoint_with_report,
    run_simulation_cuda_on_device, run_simulation_cuda_on_device_with_control,
    run_simulation_cuda_with_report,
};

#[cfg(feature = "metal")]
pub use simulation::{run_simulation_metal, run_simulation_metal_with_report};
