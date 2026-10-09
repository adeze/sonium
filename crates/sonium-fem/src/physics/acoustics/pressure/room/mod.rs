//! Room acoustics: geometrical, diffusion, and hybrid methods
//!
//! Provides tools for analysing the acoustics of enclosed spaces
//! (concert halls, studios, offices) at different frequency ranges:
//!
//! - **Ray tracing** (geometrical acoustics): high-frequency analysis where
//!   λ < room features. Computes impulse responses, energy decay curves,
//!   and standard room acoustic metrics (RT60, C80, D50, …).
//!
//! - **Acoustic diffusion**: quick energy-based analysis for large or
//!   complex spaces. Solves a diffusion equation for the acoustic energy
//!   density using the existing FEM assembler.
//!
//! - **Wave-based methods**: use the pressure acoustics time-domain or
//!   frequency-domain solvers (in the parent `acoustics` module) for
//!   low-frequency modal analysis.
//!
//! - **Hybrid**: combine wave-based (low f) with ray tracing (high f)
//!   using a crossover frequency. The FEM solution provides realistic
//!   source directivities for the ray tracer.
//!
//! # Atmosphere attenuation
//!
//! The ISO 9613-1 model computes frequency-dependent air absorption as a
//! function of temperature, pressure, and humidity. This is integrated
//! into the ray tracer and can also be used standalone.

pub mod atmosphere;
pub mod diffusion;
pub mod metrics;
pub mod ray_tracing;
