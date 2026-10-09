//! Performance and vendor-comparison harnesses for Sonium.
//!
//! This workspace owns optional CUDA vendor-library, Accelerate, OpenBLAS,
//! LAPACK, Metal, and roofline integrations.  Keep their types and linkage
//! out of the public Sonium core APIs.

pub mod compute_regression;
