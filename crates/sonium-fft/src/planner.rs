//! Native batch-aware planner facade.
//!
//! For now this just re-exports the CPU planner. Once the Metal / CUDA
//! backends land, this becomes a thin wrapper that selects the
//! appropriate concrete planner based on the `Backend` type.

pub use crate::cpu::planner::CpuFftPlanner as FftPlanner;
