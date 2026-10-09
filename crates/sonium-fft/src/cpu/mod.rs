//! CPU FFT backend.
//!
//! [`CpuFftPlanner`] caches plans by `(length, direction)`. Each plan
//! holds its own precomputed stage tables; stage construction uses quarter
//! twiddle compression where the length permits it.

pub mod plan;
pub mod planner;
