//! Differential Evolution global optimizer.
//!
//! Population-based stochastic search with multiple mutation strategies
//! (Best/Rand/Current-to-best, SHADE-style adaptive control), binomial and
//! exponential crossover, L-SHADE linear population-size reduction, an
//! external archive, Latin-hypercube / Sobol initialization, mixed-integer
//! support, and penalty-based (linear, nonlinear, equality, inequality)
//! constraint handling.
//!
//! The entry point [`differential_evolution()`] mirrors SciPy's
//! `differential_evolution` API; [`DifferentialEvolution`] exposes the
//! stateful solver. This module was merged from `math-optimisation`
//! (ported off `ndarray` onto plain vectors).

mod adaptive_config;
mod adaptive_state;
/// Integer variable handling for mixed-integer optimization.
pub mod apply_integrality;
/// Wrapper Local Search (WLS) application for local refinement.
pub mod apply_wls;
mod argmin;
mod core;
/// Binomial (uniform) crossover implementation.
pub mod crossover_binomial;
/// Exponential crossover implementation.
pub mod crossover_exponential;
mod deconfig;
mod deconfig_builder;
mod dereport;
/// SciPy-compatible [`differential_evolution()`] entry point.
pub mod differential_evolution;
/// Utilities for selecting distinct random indices from a population.
pub mod distinct_indices;
pub mod error;
/// External archive for L-SHADE algorithm.
pub mod external_archive;
/// Internal helper functions for DE implementation.
pub mod impl_helpers;
/// Latin Hypercube Sampling initialization strategy.
pub mod init_latin_hypercube;
/// Random uniform initialization strategy.
pub mod init_random;
/// Sobol quasi-random initialization strategy.
pub mod init_sobol;
mod linear_constraint_helper;
/// L-SHADE configuration.
pub mod lshade;
/// Metadata-driven optimization examples and tests.
pub mod metadata;
/// Adaptive mutation strategy with dynamic parameter control.
pub mod mutant_adaptive;
/// Best/1 mutation strategy: uses best individual plus one difference vector.
pub mod mutant_best1;
/// Best/2 mutation strategy: uses best individual plus two difference vectors.
pub mod mutant_best2;
/// Current-to-best/1 mutation: blends current with best individual.
pub mod mutant_current_to_best1;
/// Current-to-pbest/1 mutation: blends current with p-best individual (SHADE).
pub mod mutant_current_to_pbest1;
/// Rand/1 mutation strategy: uses random individual plus one difference vector.
pub mod mutant_rand1;
/// Rand/2 mutation strategy: uses random individual plus two difference vectors.
pub mod mutant_rand2;
/// Rand-to-best/1 mutation: blends random with best individual.
pub mod mutant_rand_to_best1;
mod mutation;
mod nonlinear_constraint_helper;
/// Linear penalty stacking utilities for combining multiple constraints.
pub mod stack_linear_penalty;
mod strategy;
#[cfg(test)]
mod tests;
mod types;

pub use adaptive_config::*;
pub use core::*;
pub use deconfig::*;
pub use deconfig_builder::*;
pub use dereport::*;
pub use differential_evolution::differential_evolution;
pub use error::{DEError, Result};
pub use linear_constraint_helper::*;
pub use lshade::LShadeConfig;
pub use mutation::*;
pub use nonlinear_constraint_helper::*;
pub use strategy::*;
pub use types::*;
