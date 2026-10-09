//! WaveSim: Modified Born Series Helmholtz solver on a regular grid.
//!
//! Port of [`wavesim_py`](https://github.com/IvoVellekoop/wavesim_py) — solves
//! `(∇² + k₀² ε(r)) u(r) = −s(r)` via a preconditioned Richardson iteration
//! whose preconditioner is the scaled-Laplacian Green's function evaluated in
//! Fourier space. See [`simulate`] for the high-level entry point.

pub mod analytical_1d;
pub mod boundary;
pub mod engine;
pub mod helmholtz_domain;
pub mod iteration;
pub mod source;

use num_complex::Complex64;

use crate::array::Array3;

pub use boundary::add_absorbing_boundary;
pub use engine::FftPlanCache;
pub use helmholtz_domain::{HelmholtzDomain, HelmholtzDomainError};
pub use iteration::{IterationOptions, run_richardson};
pub use source::{Source, point_source};

/// Result returned by [`simulate`].
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Computed field on the (boundary-padded) grid.
    pub u: Array3<Complex64>,
    /// Number of Richardson iterations performed.
    pub iterations: usize,
    /// Final normalized residual norm (squared, matching wavesim_py).
    pub residual_norm: f64,
    /// Pad widths `[lo, hi]` per axis so callers can crop to the physical domain.
    pub boundary_pad: [(usize, usize); 3],
}

/// High-level entry point matching `wavesim_py.simulate.simulate`.
///
/// * `permittivity` — complex `ε(r) = n²(r)` on the physical (un-padded) grid.
/// * `wavelength` and `pixel_size` share the same length unit (typically µm).
/// * `boundary_width_um` — absorbing boundary thickness (same unit) on each
///   non-periodic axis. Periodic axes get no padding.
/// * `periodic` — `[bool; 3]` matching `wavesim_py`'s `(False, True, True)` for
///   the canonical 1D example.
pub fn simulate(
    permittivity: &Array3<Complex64>,
    sources: &[Source],
    wavelength: f64,
    pixel_size: f64,
    boundary_width_um: f64,
    periodic: [bool; 3],
    options: IterationOptions,
) -> SimulationResult {
    let pad_cells = (boundary_width_um / pixel_size).ceil() as usize;
    let mut boundary_pad = [(0usize, 0usize); 3];
    for (axis, p) in periodic.iter().enumerate() {
        if !*p {
            boundary_pad[axis] = (pad_cells, pad_cells);
        }
    }

    let domain = HelmholtzDomain::new(permittivity, wavelength, pixel_size, boundary_pad, periodic);
    let padded_source = source::assemble(&domain, sources, boundary_pad);
    let (u, iterations, residual_norm) = run_richardson(&domain, &padded_source, options);

    SimulationResult {
        u,
        iterations,
        residual_norm,
        boundary_pad,
    }
}
