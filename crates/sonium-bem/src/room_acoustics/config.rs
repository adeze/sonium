//! BEM-specific configuration extensions for room acoustics simulations
//!
//! This module provides BEM-specific configuration that extends the common configuration
//! from math-xem. For the base configuration types, see sonium_xem::config.

// Re-export the common configuration types
pub use sonium_xem::{
    CrossoverConfig, DirectivityConfig, FmmConfig, FrequencyConfig, GmresConfig, IluConfig,
    MetadataConfig, Point3DConfig, RoomConfig, RoomGeometryConfig, SolverConfig, SourceConfig,
    VisualizationConfig,
};

// Any BEM-specific configuration extensions would go here
