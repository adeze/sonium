//! Radix-N butterfly codelets.
//!
//! Each codelet computes an in-place size-N DFT on N consecutive complex
//! elements. They are written as straight-line code so the compiler can
//! schedule the FMAs and keep operands in registers across the butterfly.
//!
//! These are *building blocks* — the Stockham driver in
//! [`crate::stockham`] calls them with the right twiddles in a loop.

pub mod scalar;
