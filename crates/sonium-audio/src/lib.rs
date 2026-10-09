//! Audio DSP building blocks: filters, room responses, analog color, autodiff.
//!
//! [`iir`] holds biquads, parametric EQ, SVF, Kautz systems, and IIR
//! crossovers; [`fir`] holds windowed-sinc FIR filters, FIR design from a
//! target response, and FIR crossovers. Later steps add [`dsp`], [`rir`],
//! [`analog`], and [`autodiff`].

// Rust guideline compliant 2026-02-21

pub mod analog;
pub mod autodiff;
pub mod dsp;
pub mod fir;
pub mod iir;
pub mod rir;

/// Shared numeric traits, error types, and filter helpers.
///
/// Private: [`iir`] and [`fir`] re-export everything downstream code needs.
mod common;
